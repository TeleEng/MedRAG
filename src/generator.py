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
        
        self.history = []
        
        def format_docs(docs):
            return "\n\n".join([f"[Doc {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])
            
        def translate_to_eng(query):
            print(f"\n[0] Detecting Language and Translating to English...")
            return self.translator.any_to_english(query)
            
        def translate_to_target(response, target_lang):
            print(f"\n[3] Translating English Response back to {target_lang}...")
            return self.translator.english_to_any(response, target_lang)
            
        def format_history():
            if not self.history:
                return ""
            return "".join([f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n" for q, a in self.history])

        # HyDE Chain
        hyde_prompt = PromptTemplate.from_template(
            "<|im_start|>system\nYou are a medical assistant.<|im_end|>\n{chat_history}<|im_start|>user\nWrite a brief, hypothetical medical paragraph answering this question: {question}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.hyde_chain = hyde_prompt | self.llm | StrOutputParser()
        
        def hyde_retrieval(query: str):
            print("\n[1] Generating Hypothetical Document (HyDE)...")
            hypothetical_doc = self.hyde_chain.invoke({"question": query, "chat_history": format_history()})
            expanded_query = f"{query}\n{hypothetical_doc}"
            print("\n[2] Executing Hybrid Retrieval with Expanded Query...")
            return self.retriever.invoke(expanded_query)
            
        main_prompt = PromptTemplate.from_template(
            "<|im_start|>system\nYou are a professional medical assistant. Answer the User's question using ONLY the provided context. You must cite your sources using [Doc X] format.<|im_end|>\n"
            "{chat_history}"
            "<|im_start|>user\nContext:\n{context}\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
        )
        
        self.eng_chain = (
            {
                "context": RunnableLambda(hyde_retrieval) | format_docs, 
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda _: format_history())
            }
            | main_prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.full_chain = (
            RunnableLambda(translate_to_eng)
            | (lambda x: {"eng_query": x[0], "original_lang": x[1], "eng_response": self.eng_chain.invoke(x[0])})
            | (lambda x: {"res_translated": translate_to_target(x["eng_response"], x["original_lang"]), "res_eng": x["eng_response"], "eng_query": x["eng_query"]})
        )

    def answer_query(self, raw_input: str):
        try:
            query_processed = self.input_processor.process(raw_input)
            result = self.full_chain.invoke(query_processed)
            
            # Save to memory
            self.history.append((result["eng_query"], result["res_eng"]))
            
            return result["res_translated"], result["res_eng"], []
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return "An error occurred", str(e), []

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    pipeline = MedRAGPipeline()
    # Spanish query: "What are the symptoms of diabetes?"
    query = "¿Cuáles son los síntomas de la diabetes?"
    res_translated, res_eng, docs = pipeline.answer_query(query)
    
    print("\n" + "="*50)
    print("MEDRAG TRANSLATED ANSWER:")
    print("="*50)
    print(res_translated)
    print("\n" + "="*50)
    print("ORIGINAL ENGLISH ANSWER:")
    print("="*50)
    print(res_eng)