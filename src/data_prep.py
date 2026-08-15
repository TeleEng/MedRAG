import json
from datasets import load_dataset
from src import config

def load_and_prepare_data():
    print(f"Loading dataset {config.DATASET_NAME}...")
    dataset = load_dataset(config.DATASET_NAME, split="train")
    
    # Constrain size for Kaggle
    if len(dataset) > config.MAX_SAMPLES:
        dataset = dataset.select(range(config.MAX_SAMPLES))
        
    documents = []
    training_data = []
    
    print("Processing documents and creating training pairs...")
    for item in dataset:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        # Combine instruction and input for the query context
        query = instruction
        if input_text:
            query += f" {input_text}"
            
        # The output serves as our document/chunk for the knowledge base
        if output_text and len(output_text.split()) > 10:
            documents.append({
                "id": len(documents),
                "text": output_text,
                "metadata": {"query": query}
            })
            
            # Format for instruction tuning
            training_data.append({
                "instruction": query,
                "output": output_text
            })
            
    # Save processed documents for indexing
    docs_path = config.PROCESSED_DATA_DIR / "documents.json"
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)
        
    # Save training data for QLoRA
    train_path = config.PROCESSED_DATA_DIR / "train_data.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2)
        
    print(f"Saved {len(documents)} documents to {docs_path}")
    print(f"Saved {len(training_data)} training samples to {train_path}")

if __name__ == "__main__":
    load_and_prepare_data()
