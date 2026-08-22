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
        
        self.db_type = config.GRAPH_DB_TYPE
        active_db_path = config.PROCESSED_DATA_DIR / "active_db.txt"
        
        if self.db_type == "auto":
            if active_db_path.exists():
                with open(active_db_path, "r", encoding="utf-8") as f:
                    self.db_type = f.read().strip()
            else:
                self.db_type = "kuzu"  # Safe default if not found
                
        print(f"Connecting to Graph Database ({self.db_type})...")
        
        if self.db_type == "neo4j":
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI, 
                auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            )
        elif self.db_type == "kuzu":
            import kuzu
            self.db = kuzu.Database(str(config.KUZU_DB_DIR))
            self.conn = kuzu.Connection(self.db)
        
    def __del__(self):
        if hasattr(self, 'driver') and self.db_type == "neo4j":
            self.driver.close()

    def retrieve(self, query: str):
        print("Executing Graph Retrieval...")
        graph_docs = []
        medical_keywords = ["hypertension", "diabetes", "lisinopril", "metformin", "headache", "fever", "insulin"]
        
        for kw in medical_keywords:
            if kw in query.lower():
                if self.db_type == "neo4j":
                    with self.driver.session(database="neo4j") as session:
                        result = session.run(
                            "MATCH (e:Entity {name: $name})<-[:MENTIONS]-(d:Document) "
                            "RETURN d.id as doc_id, d.text as text LIMIT 2",
                            name=kw
                        )
                        for record in result:
                            graph_docs.append({"id": record["doc_id"], "text": record["text"]})
                elif self.db_type == "kuzu":
                    # Kuzu query
                    result = self.conn.execute(
                        "MATCH (e:Entity {name: $name})<-[:MENTIONS]-(d:Document) "
                        "RETURN d.id as doc_id, d.text as text LIMIT 2",
                        {"name": kw}
                    )
                    while result.has_next():
                        record = result.get_next()
                        graph_docs.append({"id": record[0], "text": record[1]})
                        
        print("Executing FAISS and BM25 Hybrid Search...")
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
        
        # 5. Combine Graph results with Hybrid results
        all_docs = graph_docs + final_docs
        
        # Deduplicate by id
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc["id"] not in seen:
                seen.add(doc["id"])
                unique_docs.append(doc)
                
        # Return top K docs total
        return unique_docs[:config.TOP_K_RERANK]

if __name__ == "__main__":
    retriever = HybridRetriever()
    res = retriever.retrieve("What is the standard treatment for hypertension?")
    for i, r in enumerate(res):
        print(f"[{i+1}] {r['text'][:100]}...")
