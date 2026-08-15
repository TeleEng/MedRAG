import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src import config

def build_indexes():
    docs_path = config.PROCESSED_DATA_DIR / "documents.json"
    if not docs_path.exists():
        raise FileNotFoundError("Run data_prep.py first.")
        
    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
        
    texts = [doc["text"] for doc in documents]
    
    print(f"Building Dense Index for {len(texts)} documents...")
    # 1. Build FAISS Dense Index
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL_ID)
    embeddings = embed_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)
    
    faiss_path = config.PROCESSED_DATA_DIR / "faiss_index.bin"
    faiss.write_index(faiss_index, str(faiss_path))
    print(f"Dense Index saved to {faiss_path}")
    
    print("Building BM25 Sparse Index...")
    # 2. Build BM25 Sparse Index
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_path = config.PROCESSED_DATA_DIR / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 Index saved to {bm25_path}")

if __name__ == "__main__":
    build_indexes()
