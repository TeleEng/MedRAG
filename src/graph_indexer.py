import json
import os
import shutil
from src import config

class GraphIndexer:
    def __init__(self):
        self.db_type = config.GRAPH_DB_TYPE
        print(f"Initializing GraphIndexer in '{self.db_type}' mode...")
        
        if self.db_type == "neo4j":
            from neo4j import GraphDatabase
            print(f"Connecting to Neo4j at {config.NEO4J_URI}...")
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI, 
                auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            )
        elif self.db_type == "kuzu":
            import kuzu
            # Reset DB directory if it exists to avoid schema conflicts on rebuild
            if config.KUZU_DB_DIR.exists():
                shutil.rmtree(config.KUZU_DB_DIR)
            config.KUZU_DB_DIR.mkdir(parents=True, exist_ok=True)
            
            print(f"Connecting to local KùzuDB at {config.KUZU_DB_DIR}...")
            self.db = kuzu.Database(str(config.KUZU_DB_DIR))
            self.conn = kuzu.Connection(self.db)
            
            # Create schema for Kuzu (strict schema required)
            print("Creating KùzuDB Schema...")
            self.conn.execute("CREATE NODE TABLE Document (id STRING, text STRING, PRIMARY KEY (id))")
            self.conn.execute("CREATE NODE TABLE Entity (name STRING, PRIMARY KEY (name))")
            self.conn.execute("CREATE REL TABLE MENTIONS (FROM Document TO Entity)")
        else:
            raise ValueError(f"Unknown GRAPH_DB_TYPE: {self.db_type}")
            
    def close(self):
        if self.db_type == "neo4j":
            self.driver.close()
        
    def build_graph(self):
        docs_path = config.PROCESSED_DATA_DIR / "documents.json"
        if not docs_path.exists():
            raise FileNotFoundError("Run data_prep.py first.")
            
        with open(docs_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
            
        print(f"Extracting simple entities and building Graph ({self.db_type})...")
        medical_keywords = ["hypertension", "diabetes", "lisinopril", "metformin", "headache", "fever", "insulin"]
        
        if self.db_type == "neo4j":
            self._build_neo4j(documents, medical_keywords)
        elif self.db_type == "kuzu":
            self._build_kuzu(documents, medical_keywords)
            
        print("Graph Indexing Complete!")

    def _build_neo4j(self, documents, medical_keywords):
        with self.driver.session(database="neo4j") as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            
            for doc in documents:
                doc_id = doc["id"]
                text = doc["text"].lower()
                
                session.run(
                    "MERGE (d:Document {id: $id}) "
                    "SET d.text = $text",
                    id=doc_id, text=doc["text"]
                )
                
                for keyword in medical_keywords:
                    if keyword in text:
                        session.run(
                            """
                            MERGE (e:Entity {name: $name})
                            MERGE (d:Document {id: $doc_id})
                            MERGE (d)-[:MENTIONS]->(e)
                            """,
                            name=keyword, doc_id=doc_id
                        )

    def _build_kuzu(self, documents, medical_keywords):
        # Kuzu syntax: CREATE or MERGE works similarly, but we use MERGE for idempotency
        for doc in documents:
            doc_id = doc["id"]
            text = doc["text"].lower()
            
            # Insert document
            self.conn.execute(
                "MERGE (d:Document {id: $id}) ON CREATE SET d.text = $text",
                {"id": doc_id, "text": doc["text"]}
            )
            
            for keyword in medical_keywords:
                if keyword in text:
                    self.conn.execute(
                        "MERGE (e:Entity {name: $name})",
                        {"name": keyword}
                    )
                    self.conn.execute(
                        "MATCH (d:Document {id: $doc_id}), (e:Entity {name: $name}) "
                        "MERGE (d)-[:MENTIONS]->(e)",
                        {"doc_id": doc_id, "name": keyword}
                    )

if __name__ == "__main__":
    indexer = GraphIndexer()
    indexer.build_graph()
    indexer.close()
