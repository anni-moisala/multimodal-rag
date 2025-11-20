# Multimodal RAG using ColQwen and Qdrant

This project implements **multimodal RAG** incorporating both text and visual contents from document page images.

Retrieval is done using [ColQwen2](https://huggingface.co/vidore/colqwen2-v1.0-hf), a document retrieval model based on [ColPali](https://huggingface.co/blog/manu/colpali), which uses vision LLMs + late interaction to embed and retrieve image representation of document pages, preserving full visual content.

The embedding are stored in a [Qdrant](https://qdrant.tech/) vector database on disk. Qdrant is an open source vector database that supports multi vector (late interaction) embeddings created by ColQwen.

Finally, the retrieved images are given to a [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) vision language model for answer generation.

## Data

We use a subset of the [RAG-60K](https://github.com/CSCfi/RAG-60K) copernicus dataset.

## Multi-GPU support

Generating multi vector embeddings for thousands of PDF image pages can take hours. This project uses Torch Distributed Data Parallel (DDP) to utilize one full node (8 GPUs) on LUMI to speed things up.
Embeddings are gathered on one GPU for uploading to Qdrant. This pipeline has been tested on 10K PDFs. 

## How to run on LUMI

The CSC preinstalled PyTorch module covers most of the libraries needed to run these examples. The rest can be installed on top of the module in a virtual environment. We also load an image with poppler needed to convert pdf pages to images. 

### Load the module
```bash
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7
```
### Create and activate a virtual environment using system packages
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```
### Install packages
```bash
pip install PyPDF2 qdrant-client --cache-dir ./.pip-cache 
```

### Upgrade transformers
```bash
pip install --upgrade transformers --cache-dir ./.pip-cache 
```
The flag --cache-dir points the pip cache to the current (scratch) folder instead of the default (home directory), to avoid filling up home directory quota. 

### Run ingest.py
Edit the source folder path to point to your data in `ingest.py`, you can use your own data or follow instructions in [RAG-60K](https://github.com/CSCfi/RAG-60K) to download the copernicus dataset.
Then edit the `run_ingest.sh` batch job script project number and venv path, as well as the requested allocations to match the size of your dataset. Then run

```bash
sbatch run_ingest.sh
```

### Run query.py

Run  `query.py` interactively by requesting resources (change your project code).

```bash
srun --account=project_462000824 --partition=dev-g --ntasks=1 --cpus-per-task=7 --gpus-per-node=1 --mem=120G  --time=01:00:00 --nodes=1 --pty bash 
```

Load module, load container, set HF cache dir and activate venv.

```bash
module purge                                                                                                                 
module use /appl/local/csc/modulefiles/                                                                                      
module load pytorch/2.7                                                                                                      
                                                                                                                             
export SING_IMAGE=/pfs/lustrep4/appl/local/csc/soft/ai/images/pytorch_2.7.1_lumi_vllm-0.8.5.post1.sif                        
export HF_HOME=/scratch/project_462000824/cache/hf/hub
export HF_HUB_ENABLE_HF_TRANSFER=1

source venv/bin/activate
```
Then, run the script.

```bash
python3 query.py
```

## Notes

Loading the Qdrant client for querying from 10k PDFs took very long (80mins).
However, search is fast due to HNSW indexing, and qdrant offers for example built in quantization for further speedups.  
Also check out this alternative implementation that does not require any external package for a vector database: https://github.com/shanshanwangcsc/Multimodal-RAG/tree/main.
