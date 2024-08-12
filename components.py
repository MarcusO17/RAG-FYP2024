import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
#from llama_index.llms.groq import Groq
from groqllm import GroqLLM
from llama_index.core import PromptTemplate
from dotenv import load_dotenv
import os
import chromadb
class Components:
    load_dotenv()

    def __init__(self,embed,reranker,model_name):
        self.embed = embed
        self.reranker = reranker
        self.model_name = model_name

    def get_embedding_model(self):
        embedding_model = HuggingFaceEmbedding(model_name = self.embed)
        print("Embedding model loaded!")
        return embedding_model

    def get_reranker(self):
        reranker_model = FlagEmbeddingReranker(model=self.reranker, top_n=5)
        print("Reranker model loaded!")
        return reranker_model
    
    def get_db(self):
        db = chromadb.EphemeralClient()
        chroma_collection = db.get_or_create_collection("temp")
        return chroma_collection
    
    def get_groq_llm(self):
        return GroqLLM(self.model_name,api_key=os.getenv("GROQ_API_KEY"),temperature=0.1)
        
    def get_qa_template(self):
        qa_template_str = """
        Context: {context_str}
        Instructions:
        - Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
        - Utilize the context provided for accurate and specific information.
        - Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
        - Be concise and to the point.
        Question: {query_str}
        """
        if self.model_name == "llama-3.1-70b-versatile":
           return PromptTemplate(qa_template_str)
        elif self.model_name == "llama-3.1-8b-instant":
           return PromptTemplate(qa_template_str)
        elif self.model_name == " llama3-70b-8192":
           return PromptTemplate(qa_template_str)
        elif self.model_name == "llama3-8b-8192":
            return PromptTemplate(qa_template_str)
        elif self.model_name == "mixtral-8x7b-32768":
            return PromptTemplate(qa_template_str)
        elif self.model_name == "gemma2-9b-it":
           return PromptTemplate(qa_template_str)

