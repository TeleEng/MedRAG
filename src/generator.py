import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.retriever import HybridRetriever
from src.translator import MedTranslator
from src import config

class MedRAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.translator = MedTranslator()
        
        print("Loading Qwen Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(config.MODELS_DIR / "medrag_adapter"))
        
        print("Loading Base Model for Generation...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID,
            torch_dtype=torch.float16, # Run fast in fp16
            device_map="auto"
        )
        
        print("Loading trained QLoRA adapter...")
        self.model = PeftModel.from_pretrained(
            base_model,
            str(config.MODELS_DIR / "medrag_adapter")
        )
        self.model.eval()

    def answer_query(self, query_persian: str):
        print(f"\n[0] Translating User Query from Persian to English...")
        query_english = self.translator.persian_to_english(query_persian)
        print(f"Translated Query: '{query_english}'")
        
        print(f"\n[1] Retrieving Graph and FAISS context for: '{query_english}'")
        retrieved_docs = self.retriever.retrieve(query_english)
        
        context_str = "\n\n".join([f"[Doc {i+1}]: {doc['text']}" for i, doc in enumerate(retrieved_docs)])
        
        system_prompt = (
            "You are a professional medical assistant. "
            "Answer the User's question using ONLY the provided context. "
            "You must cite your sources using [Doc X] format."
        )
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query_english}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        
        print(f"[2] Generating response in English...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.15,
                do_sample=True,
            )
            
        prompt_length = inputs.input_ids.shape[1]
        response_english = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        print(f"English Response Generated.")
        
        print(f"\n[3] Translating English Response back to Persian...")
        response_persian = self.translator.english_to_persian(response_english)
        
        return response_persian, response_english, retrieved_docs

if __name__ == "__main__":
    pipeline = MedRAGPipeline()
    # "What are the symptoms of diabetes?" in Persian
    query_pes = "علائم بیماری دیابت چیست؟" 
    res_pes, res_eng, docs = pipeline.answer_query(query_pes)
    
    print("\n" + "="*50)
    print("MEDRAG PERSIAN ANSWER:")
    print("="*50)
    print(res_pes)
    print("\n" + "="*50)
    print("ORIGINAL ENGLISH ANSWER:")
    print("="*50)
    print(res_eng)
