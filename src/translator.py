import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src import config
import langdetect

# Map ISO 639-1 (langdetect) to FLORES-200 (NLLB)
LANG_MAP = {
    "en": "eng_Latn",
    "fa": "pes_Arab",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "zh-cn": "zho_Hans",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "tr": "tur_Latn",
    "hi": "hin_Deva",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang"
}

class MedTranslator:
    def __init__(self):
        print("Loading Translation Module (NLLB-200)...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.TRANSLATOR_MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.TRANSLATOR_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def detect_lang(self, text: str) -> str:
        try:
            detected = langdetect.detect(text)
            return LANG_MAP.get(detected, "eng_Latn") # Default to English if unsupported
        except:
            return "eng_Latn"

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if src_lang == tgt_lang:
            return text
            
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

    def any_to_english(self, text: str) -> tuple:
        detected_lang = self.detect_lang(text)
        print(f"Detected Language: {detected_lang}")
        translated = self.translate(text, src_lang=detected_lang, tgt_lang="eng_Latn")
        return translated, detected_lang

    def english_to_any(self, text: str, target_lang: str) -> str:
        return self.translate(text, src_lang="eng_Latn", tgt_lang=target_lang)