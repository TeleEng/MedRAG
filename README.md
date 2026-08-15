# MedRAG: Domain-Expert Medical Q&A System

MedRAG is an end-to-end instruction-tuned Retrieval-Augmented Generation (RAG) pipeline designed specifically for the medical domain. It combines semantic search, lexical search, cross-encoder reranking, and QLoRA fine-tuning to deliver highly accurate, grounded, and professionally formatted medical answers.

## Features
- **Hybrid Retrieval**: Combines FAISS (Dense) and BM25 (Lexical) using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Re-orders retrieved chunks for maximum context precision.
- **QLoRA Fine-Tuning**: Instruction-tunes a base LLM (e.g., Qwen2.5-1.5B) using 4-bit NormalFloat quantization.
- **Kaggle-Ready**: Designed to train and evaluate within the compute limits of a Kaggle Notebook (dual T4 GPUs).

## Project Structure
```text
MedRAG/
├── data/               # Raw and processed datasets, FAISS indexes
├── src/
│   ├── config.py       # Global paths and hyperparameters
│   ├── data_prep.py    # Dataset downloading and semantic chunking
│   ├── indexer.py      # FAISS and BM25 index creation
│   ├── retriever.py    # Hybrid search and reranking logic
│   ├── trainer.py      # QLoRA fine-tuning using SFTTrainer
│   └── generator.py    # End-to-end generation and inference
├── train_kaggle.ipynb  # Executable notebook for Kaggle environment
└── requirements.txt    # Python dependencies
```

## Getting Started on Kaggle
1. Open a new Kaggle Notebook and set the Accelerator to **GPU T4 x2**.
2. Upload this repository or clone it directly inside the notebook.
3. Run the `train_kaggle.ipynb` notebook from top to bottom.
