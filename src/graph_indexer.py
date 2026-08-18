import json
from neo4j import GraphDatabase
from src import config

class GraphIndexer:
    def __init__(self):
        print(f"Connecting to Neo4j at {config.NEO4J_URI}...")
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, 
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        
    def close(self):
        self.driver.close()
        
    def build_graph(self):
        docs_path = config.PROCESSED_DATA_DIR / "documents.json"
        if not docs_path.exists():
            raise FileNotFoundError("Run data_prep.py first.")
            
        with open(docs_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
            
        print("Extracting simple entities and building Graph...")
        
        # A simple keyword-based extraction strategy for demo purposes.
        # In a real production system, you would use GLiNER or SpaCy here.
        medical_keywords = ["hypertension", "diabetes", "lisinopril", "metformin", "headache", "fever", "insulin"]
        
        with self.driver.session(database="neo4j") as session:
            # Create constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            
            for doc in documents:
                doc_id = doc["id"]
                text = doc["text"].lower()
                
                # Insert the Document node
                session.run(
                    "MERGE (d:Document {id: $id}) "
                    "SET d.text = $text",
                    id=doc_id, text=doc["text"]
                )
                
                # Extract and link entities
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
                        
        print("Graph Indexing Complete!")

if __name__ == "__main__":
    indexer = GraphIndexer()
    indexer.build_graph()
    indexer.close()
