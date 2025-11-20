# Multimodal RAG using ColQwen and Qdrant

This project implements multimodal RAG incorporating both text and visual contents from document page images.

Retrieval is done using [ColQwen](https://huggingface.co/vidore/colqwen2-v1.0-hf), a document retrieval model based on [ColPali](https://huggingface.co/blog/manu/colpali), which uses vision LLMs + late interaction to embed and retrieve image representation of document pages, preserving full visual content.

The embedding are stored in a self hosted on disk [Qdrant](https://qdrant.tech/) vector database. Qdrant is an open source vector database that supports multi vector (late interaction) embeddings created by ColQwen.

Finally, the retrieved images are given to a [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) vision language model for answer generation.

## Data

We use a subset of the [RAG-60K](https://github.com/CSCfi/RAG-60K) copernicus dataset.

## Multi-GPU support

Generating multi vector embeddings for thousands of PDF image pages can take hours. This project uses Torch Distributed Data Parallel (DDP) to utilize one full node (8 GPUs) on LUMI to speed things up.
Embeddings are gathered on one GPU for uploading to Qdrant. This pipeline has been tested on 10K PDFs. 

## Notes

- Loading the Qdrant client for querying from 10k PDFs took very long (80mins), not making this approach suitable for very large datasets. 
