import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset Config
DATASET_NAME = "medalpaca/medical_meadow_wikidoc"
MAX_SAMPLES = 10000  # Constrain for Kaggle environment

# Chunking Config
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval Config
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVE = 10
TOP_K_RERANK = 3
RRF_K = 60

# Model & Training Config
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
EPOCHS = 1
