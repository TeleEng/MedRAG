import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from src import config

class HybridRetriever:
    def __init__(self):
        print("Loading documents and indexes...")
        with open(config.PROCESSED_DATA_DIR / "documents.json", "r", encoding="utf-8") as f:
            self.documents = json.load(f)
            
        self.faiss_index = faiss.read_index(str(config.PROCESSED_DATA_DIR / "faiss_index.bin"))
        
        with open(config.PROCESSED_DATA_DIR / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
            
        self.bi_encoder = SentenceTransformer(config.EMBEDDING_MODEL_ID)
        self.cross_encoder = CrossEncoder(config.CROSS_ENCODER_ID)
        
    def retrieve(self, query: str):
        # 1. Dense Retrieval
        q_emb = self.bi_encoder.encode([query], normalize_embeddings=True).astype("float32")
        dense_scores, dense_ids = self.faiss_index.search(q_emb, config.TOP_K_RETRIEVE)
        dense_ids = dense_ids[0].tolist()
        
        # 2. Sparse Retrieval
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ids = np.argsort(bm25_scores)[::-1][:config.TOP_K_RETRIEVE].tolist()
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, doc_id in enumerate(dense_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (config.RRF_K + rank + 1)
            
        for rank, doc_id in enumerate(bm25_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (config.RRF_K + rank + 1)
            
        merged_ids = [doc_id for doc_id, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:config.TOP_K_RETRIEVE]]
        
        # 4. Cross-Encoder Reranking
        candidate_docs = [self.documents[doc_id] for doc_id in merged_ids]
        pairs = [[query, doc["text"]] for doc in candidate_docs]
        ce_scores = self.cross_encoder.predict(pairs)
        
        reranked = sorted(zip(candidate_docs, ce_scores), key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, score in reranked][:config.TOP_K_RERANK]
        
        return final_docs

if __name__ == "__main__":
    retriever = HybridRetriever()
    res = retriever.retrieve("What is the standard treatment for hypertension?")
    for i, r in enumerate(res):
        print(f"[{i+1}] {r['text'][:100]}...")
