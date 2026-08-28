import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from src import config

def train_model():
    print("Loading tokenizer and configuring formatting...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading QLoRA configuration...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading Base Model: {config.BASE_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    print("Loading training data...")
    dataset = load_dataset("json", data_files=str(config.PROCESSED_DATA_DIR / "train_data.json"), split="train")
    
    # We define a formatting function for SFTTrainer
    def formatting_prompts_func(example):
        if isinstance(example.get('instruction'), list):
            output_texts = []
            for i in range(len(example['instruction'])):
                text = f"User: {example['instruction'][i]}\n\nAssistant: {example['output'][i]}"
                output_texts.append(text)
            return output_texts
        else:
            return f"User: {example['instruction']}\n\nAssistant: {example['output']}"
        
    training_args = SFTConfig(
        output_dir=str(config.MODELS_DIR / "checkpoints"),
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        optim="paged_adamw_8bit",
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="cosine",
        max_steps=50, # Set to small number just for Kaggle demo purposes
        logging_steps=10,
        fp16=True, # Use fp16 for T4 compatibility, bf16 for Ampere+
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        max_seq_length=config.MAX_SEQ_LENGTH,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )
    
    print("Starting QLoRA Fine-Tuning...")
    trainer.train()
    
    print("Saving trained adapter...")
    trainer.model.save_pretrained(str(config.MODELS_DIR / "medrag_adapter"))
    tokenizer.save_pretrained(str(config.MODELS_DIR / "medrag_adapter"))
    print("Training complete! Run generator.py for end-to-end RAG.")

if __name__ == "__main__":
    train_model()
