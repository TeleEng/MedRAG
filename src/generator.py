import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
from peft import PeftModel
from src.retriever import HybridRetriever, LangChainMedRAGRetriever
from src.translator import MedTranslator
from src.data_prep import TextInputProcessor
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src import config

class MedRAGPipeline:
    def __init__(self):
        self.retriever = LangChainMedRAGRetriever()
        self.translator = MedTranslator()
        self.input_processor = TextInputProcessor()
        
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
        
        pipe = hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=300,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        def format_docs(docs):
            return "\n\n".join([f"[Doc {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])
            
        def translate_to_eng(query):
            print(f"\n[0] Translating User Query to English...")
            return self.translator.persian_to_english(query)
            
        def translate_to_pes(response):
            print(f"\n[3] Translating English Response back to Persian...")
            return self.translator.english_to_persian(response)
            
        prompt = PromptTemplate.from_template(
            "<|im_start|>system\nYou are a professional medical assistant. Answer the User's question using ONLY the provided context. You must cite your sources using [Doc X] format.<|im_end|>\n"
            "<|im_start|>user\nContext:\n{context}\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
        )
        
        self.eng_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.full_chain = (
            RunnableLambda(translate_to_eng)
            | (lambda q: {"eng_query": q, "eng_response": self.eng_chain.invoke(q)})
            | (lambda x: {"res_pes": translate_to_pes(x["eng_response"]), "res_eng": x["eng_response"]})
        )

    def answer_query(self, raw_input: str):
        try:
            query_persian = self.input_processor.process(raw_input)
            result = self.full_chain.invoke(query_persian)
            return result["res_pes"], result["res_eng"], []
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return "An error occurred", str(e), []

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
