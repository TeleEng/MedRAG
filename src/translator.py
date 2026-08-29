import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src import config

class MedTranslator:
    def __init__(self):
        print("Loading Translation Module (NLLB-200)...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.TRANSLATOR_MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.TRANSLATOR_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=512
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def persian_to_english(self, text: str) -> str:
        return self.translate(text, src_lang="pes_Arab", tgt_lang="eng_Latn")

    def english_to_persian(self, text: str) -> str:
        return self.translate(text, src_lang="eng_Latn", tgt_lang="pes_Arab")
