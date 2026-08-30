# MedRAG: Cross-Lingual GraphRAG Medical Q&A System

MedRAG is an end-to-end instruction-tuned Retrieval-Augmented Generation (RAG) pipeline designed specifically for the medical domain. It combines state-of-the-art **Graph Retrieval**, **Hybrid Semantic/Lexical Search**, **Advanced Retrieval Strategies (HyDE)**, **Cross-Encoder Reranking**, and **Cross-Lingual Translation** to deliver highly accurate, grounded, and professionally formatted medical answers in multiple languages.

## 🌟 Advanced Pipeline Architecture

The MedRAG architecture is orchestrated using **LangChain** (LCEL), creating a robust, modular, and highly complex pipeline:

1. **Modular Input Processing**: Accepts inputs via a generalized `InputProcessor` interface, future-proofed for Multimodal/OCR medical document ingesting.
2. **Cross-Lingual Translation**: Leverages Facebook's `nllb-200-distilled-600M` to intercept queries in any language (Persian, Spanish, French, Arabic, Chinese, etc.) using `langid`, translate them to English for exact retrieval and reasoning, and translate the final generated answer back to the user's native language.
3. **Conversational Memory**: Maintains multi-turn context via dynamic memory injection, allowing the LLM to remember previous medical queries and follow-up questions.
4. **Hypothetical Document Embeddings (HyDE)**: Before searching the database, MedRAG uses the LLM to generate a hypothetical, "fake" medical answer. This text is appended to the query to radically improve the dense semantic matching.
5. **Entity Extraction (NER)**: Uses `SpaCy` (`en_core_web_md`) to dynamically extract Named Entities (Diseases, Chemicals, etc.) from the expanded query.
6. **Triple-Vector Retrieval Engine**:
   - **Graph Database (Neo4j & Kùzu)**: Executes Cypher queries against the extracted entities to find deterministic document nodes. Features a robust auto-fallback from Cloud Neo4j AuraDB to an embedded local `KùzuDB` if network connectivity drops.
   - **Dense Retrieval (FAISS)**: Performs semantic search using sentence-transformers (`all-MiniLM-L6-v2`).
   - **Sparse Retrieval (BM25)**: Performs keyword-based lexical search.
7. **Reciprocal Rank Fusion (RRF)**: Merges the FAISS and BM25 results mathematically.
8. **Cross-Encoder Reranking**: Passes the merged candidate documents through an MS-MARCO Cross-Encoder to re-order them for maximum contextual precision.
9. **QLoRA Generation**: Generates the final grounded response using a 4-bit Quantized, Instruction-Tuned `Qwen/Qwen2.5-1.5B-Instruct` model, wrapped seamlessly into a `HuggingFacePipeline`.
10. **Automated Evaluation (RAGAS)**: Integrates RAGAS to programmatically evaluate the LLM's answers for hallucinations (Faithfulness) using the local Qwen model as the evaluator.

## 🛠️ Project Structure
```text
MedRAG/
├── data/                  # Raw and processed datasets, FAISS indexes
├── src/
│   ├── config.py          # Global paths, credentials, and hyperparameters
│   ├── data_prep.py       # Dataset downloading, chunking, and InputProcessors
│   ├── indexer.py         # FAISS and BM25 index creation
│   ├── graph_indexer.py   # SpaCy NER Extraction & Graph Building (Neo4j/Kùzu)
│   ├── retriever.py       # Hybrid retrieval and LangChain BaseRetriever wrapper
│   ├── translator.py      # Cross-Lingual NLLB-200 Translation Module
│   ├── trainer.py         # QLoRA fine-tuning with OOM exception handling
│   └── generator.py       # LCEL Orchestration (Memory, HyDE, Pipeline)
├── train_kaggle.ipynb     # Executable notebook for Kaggle environment
└── requirements.txt       # Python dependencies
```

## 🚀 Getting Started on Kaggle

Since MedRAG supports remote Graph hosting, you can configure your Neo4j credentials securely using **Kaggle Secrets**. (If you skip this, it will simply fall back to local KùzuDB seamlessly!)

1. Open a new Kaggle Notebook and set the Accelerator to **GPU T4 x2**.
2. Go to **Add-ons** > **Secrets** in the top menu and add your AuraDB credentials:
   - `NEO4J_URI` (e.g., `neo4j+s://<your-instance-id>.databases.neo4j.io`)
   - `NEO4J_USERNAME` (usually `neo4j`)
   - `NEO4J_PASSWORD`
3. Toggle the switch to **Attach** the secrets to your notebook.
4. Paste the setup cell from `train_kaggle.ipynb` into your notebook. This script explicitly resets the directory and performs a clean wipe-and-clone so you can re-run the notebook repeatedly without kernel restarts.
5. Run the rest of the notebook from top to bottom.

## 💻 Local Setup

If running locally instead of Kaggle, you can create a `.env` file in the root directory:
```env
NEO4J_URI=neo4j+s://<your-instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
```
*(If no `.env` file is provided, the local KùzuDB fallback will activate automatically).*