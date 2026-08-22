# MedRAG: Cross-Lingual GraphRAG Medical Q&A System

MedRAG is an end-to-end instruction-tuned Retrieval-Augmented Generation (RAG) pipeline designed specifically for the medical domain. It combines state-of-the-art **Graph Retrieval (Neo4j)**, **Hybrid Semantic/Lexical Search**, **Cross-Encoder Reranking**, and **Cross-Lingual Translation** to deliver highly accurate, grounded, and professionally formatted medical answers in multiple languages.

## Features
- **GraphRAG (Neo4j)**: Extracts medical entities and establishes a deterministic Knowledge Graph hosted on Neo4j AuraDB to enhance retrieval.
- **Cross-Lingual Translation**: Leverages Facebook's `nllb-200-distilled-600M` to intercept Persian queries, translate them to English for exact retrieval and reasoning, and translate the final generated answer back to Persian.
- **Hybrid Retrieval**: Combines FAISS (Dense) and BM25 (Lexical) using Reciprocal Rank Fusion (RRF) alongside Neo4j Cypher queries.
- **Cross-Encoder Reranking**: Re-orders retrieved chunks for maximum context precision.
- **QLoRA Fine-Tuning**: Instruction-tunes a base LLM (`Qwen/Qwen2.5-1.5B-Instruct`) using 4-bit NormalFloat quantization.
- **Kaggle-Ready**: Designed to train and evaluate within the compute limits of a Kaggle Notebook (dual T4 GPUs) seamlessly connecting with Kaggle Secrets.

## Project Structure
```text
MedRAG/
├── data/                  # Raw and processed datasets, FAISS indexes
├── src/
│   ├── config.py          # Global paths, credentials, and hyperparameters
│   ├── data_prep.py       # Dataset downloading and semantic chunking
│   ├── indexer.py         # FAISS and BM25 index creation
│   ├── graph_indexer.py   # Neo4j Entity Extraction and Graph Building
│   ├── retriever.py       # Graph + Hybrid search and reranking logic
│   ├── translator.py      # Cross-Lingual NLLB-200 Translation Module
│   ├── trainer.py         # QLoRA fine-tuning using SFTTrainer
│   └── generator.py       # End-to-end generation and inference
├── train_kaggle.ipynb     # Executable notebook for Kaggle environment
└── requirements.txt       # Python dependencies
```

## Getting Started on Kaggle

Since MedRAG relies on a remote Knowledge Graph, you must configure your Neo4j credentials securely using **Kaggle Secrets**.

1. Open a new Kaggle Notebook and set the Accelerator to **GPU T4 x2**.
2. Go to **Add-ons** > **Secrets** in the top menu and add your AuraDB credentials:
   - `NEO4J_URI` (e.g., `neo4j+s://<your-instance-id>.databases.neo4j.io`)
   - `NEO4J_USERNAME` (usually `neo4j`)
   - `NEO4J_PASSWORD`
3. Toggle the switch to **Attach** the secrets to your notebook.
4. Clone this repository directly inside the notebook by running the first cell of `train_kaggle.ipynb`.
5. Run the rest of the notebook from top to bottom.

## Local Setup

If running locally instead of Kaggle, simply create a `.env` file in the root directory:
```env
NEO4J_URI=neo4j+s://<your-instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
```
