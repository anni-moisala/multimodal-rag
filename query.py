import os
import time
import torch
from tqdm import tqdm
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from torch.utils.data import Dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import (
    AutoProcessor,
    AutoModelForPreTraining,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
)
from transformers.utils.import_utils import is_flash_attn_2_available
from qwen_vl_utils import process_vision_info
from ingest_qdrant import PDFImagesDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source_folder = "/scratch/project_462000824/amoisala/RAG-60K/data/copernicus"
generation_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
retrieval_model_name = "vidore/colqwen2-v1.0-hf"                                
index_path = "/flash/project_462000824/amoisala/qdrant_index_1000"
collection_name = "qdrant_index"
output_dir = "retrieved_images"                                                            
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()
qdrant_client = QdrantClient(path=index_path)
end_time = time.time()
print(f"Index setup time: {end_time - start_time:.2f}s")  

generation_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(generation_model_name)
generation_processor = AutoProcessor.from_pretrained(generation_model_name)

# Set image size
min_pixels = 256*28*28
max_pixels = 1024*28*28 

# For retrieval
retrieval_processor = AutoProcessor.from_pretrained(
    retrieval_model_name,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

retrieval_model = AutoModelForPreTraining.from_pretrained(
    retrieval_model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    ignore_mismatched_sizes=True, # to supress warnings
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).to(device)

def search_images_by_text(query_text, top_k):
    with torch.no_grad():
        inputs_text = retrieval_processor(text=[query_text]).to(retrieval_model.device)
        query_embeddings = retrieval_model(**inputs_text).embeddings

    multivector_query = query_embeddings[0].cpu().float().numpy().tolist()

    start_time = time.time()
    search_result = qdrant_client.query_points(
        collection_name=collection_name, query=multivector_query, limit=top_k
    )
    elapsed = time.time() - start_time
    print(f"Total query time: {elapsed:.3f} seconds")

    return search_result
                                
def answer_with_multimodal_rag(model, processor, query_text, top_k, max_new_tokens):
    results = search_images_by_text(query_text, top_k=top_k)

    images = []
    start_time = time.time() 
    dataset = PDFImagesDataset(source_folder)
    for rank, point in enumerate(results.points, start=1):
        pdf_path = point.payload["pdf_path"]
        page_index = point.payload["page_index"]
        images_from_pdf = convert_from_path(pdf_path, first_page=page_index+1, last_page=page_index+1)
        image = images_from_pdf[0]
        save_path = os.path.join(output_dir, f"{query_text.replace(' ', '_')}_result_{rank}.png")
        image.save(save_path)
        print(f"Saved image {rank} to {save_path}")
        print("Retrieved page ", page_index)
        images.append(image)

    end_time = time.time()                                                       
    elapsed = end_time - start_time
    print(f"Total image retrieval time: {elapsed:.3f} seconds")

    chat_template = [
    {
      "role": "user",
      "content": [
          {"type": "image", "image": image} for image in images
            ] + [
          {"type": "text", "text": query_text}
        ],
      }
    ]

    # Prepare the inputs
    text = generation_processor.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(chat_template)
    inputs = generation_processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(generation_model.device)

    start_time = time.time() 
    # Generate text from the vl_model
    generated_ids = generation_model.generate(**inputs, max_new_tokens=max_new_tokens)
    end_time = time.time()                                                       
    elapsed = end_time - start_time
    print(f"Total vl model inference time: {elapsed:.3f} seconds")

    # Postprocessing
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text, results

def chat_loop():
    #print("Enter 'exit' to stop.")
    while True:
        query_text = input("\nQuery: ")
        if query_text.lower() in ('exit'):
            print("\nGoodbye!")
            break
        
        output_text, results = answer_with_multimodal_rag(
            model=generation_model,
            processor=generation_processor,
            query_text=query_text,
            top_k=3,
            max_new_tokens=512
        )
        
        print("\nAnswer:")
        print(output_text[0])

chat_loop()
