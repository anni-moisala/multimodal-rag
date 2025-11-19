import os
import torch
import time
import io
import numpy as np
from tqdm import tqdm
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from transformers import ColQwen2ForRetrieval, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available

from qdrant_client import QdrantClient
from qdrant_client.http import models

source_folder = "/scratch/project_462000824/amoisala/RAG-60K/data/copernicus"
model_name = "vidore/colqwen2-v1.0-hf"
index_path = "./qdrant_index_10k"
collection_name = "qdrant_index_10k"
os.makedirs(index_path, exist_ok=True)

batch_size = 8
num_workers = 4

class PDFImagesDataset(Dataset):
    def __init__(self, pdf_folder):
        self.pages = []
        self.pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")][:1000]
        for pdf_file in self.pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            try:
                reader = PdfReader(pdf_path)
                for page_index in range(len(reader.pages)):
                    self.pages.append((pdf_path, page_index))
            except Exception as e:
                print(f"Error reading {pdf_path}: {e}")
        print(f"Loaded {len(self.pages)} pages from {len(self.pdf_files)} PDFs")

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, idx):
        pdf_path, page_index = self.pages[idx]
        image = convert_from_path(pdf_path, first_page=page_index + 1, last_page=page_index + 1)[0]
        return {"image": image, "idx": idx}

def collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    indices = [item["idx"] for item in batch]
    processed = processor.process_images(images)
    processed["indices"] = torch.tensor(indices)
    return processed

def gather_and_upload(local_embeddings, local_indices, dataset, qdrant_client=None):
    """Gather variable-length embeddings from all DDP ranks and upload to Qdrant."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Convert CPU float32 tensors to NumPy arrays
    local_embeddings_cpu = [emb.numpy() for emb in local_embeddings]

    # Serialize embeddings to bytes for variable-length all_gather
    serialized_bytes = []
    for emb in local_embeddings_cpu:
        buf = io.BytesIO()
        np.save(buf, emb)
        serialized_bytes.append(buf.getvalue())

    # Prepare gather lists
    gathered_bytes = [None for _ in range(world_size)]
    gathered_indices = [None for _ in range(world_size)]

    # all_gather variable-length embeddings and indices
    dist.all_gather_object(gathered_bytes, serialized_bytes)
    dist.all_gather_object(gathered_indices, local_indices)

    if rank == 0:
        # reconstruct embeddings
        all_embeddings = [
            np.load(io.BytesIO(b)) for sublist in gathered_bytes for b in sublist
        ]
        all_indices = [i for sublist in gathered_indices for i in sublist]

        # build Qdrant points
        points = []
        for idx, emb in zip(all_indices, all_embeddings):
            pdf_path, page_index = dataset.pages[idx]
            points.append(
                models.PointStruct(
                    id=int(idx),
                    vector=emb.tolist(),  # Qdrant expects flat list[float]
                    payload={"pdf_path": pdf_path, "page_index": int(page_index)}
                )
            )

        # upload all points at once
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False  # set to True if you want synchronous upload
        )

    dist.barrier()

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Load model & processor
    model = ColQwen2ForRetrieval.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
    ).to(device)
    processor = ColQwen2Processor.from_pretrained(model_name)
    model = DDP(model, device_ids=[local_rank])
    model.eval()
    # Dataset & DataLoader
    dataset = PDFImagesDataset(source_folder)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    # Only rank 0 creates the collection
    if rank == 0:
        qdrant_client = QdrantClient(path=index_path)
        if not qdrant_client.collection_exists(collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                on_disk_payload=True,
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=100),
                vectors_config=models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            )
    dist.barrier()  # Ensure collection exists before other ranks continue

    start_time = time.time()
    for batch in tqdm(dataloader, desc=f"Rank {rank}"):
        indices = batch.pop("indices").to("cpu").tolist()
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            emb = model(**batch).embeddings # float16 on gpu

        # Convert embeddings to CPU float32 and unbind into list
        batch_embeddings = list(torch.unbind(emb.to("cpu").float()))

        gather_and_upload(batch_embeddings, indices, dataset, qdrant_client if rank == 0 else None)
        #print("processed a batch")

    print("Indexing complete!")
    end_time = time.time()
    print(f"Total Indexing time: {end_time - start_time:.2f}s")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
