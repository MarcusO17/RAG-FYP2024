import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.llms.groq import Groq
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
        return Groq(self.model_name,api_key=os.getenv("GROQ_API_KEY"))
        
    def get_qa_template(self):
        if self.model_name == "llama-3.1-70b-versatile":
            return """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {query_str} 
            Context: {context_str} 
            Answer:"""
        elif self.model_name == "llama-3.1-8b-instant":
            return """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {query_str} 
            Context: {context_str}
            Answer:"""
        elif self.model_name == " llama3-70b-8192":
            return """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {query_str}  
            Context: {context_str}
            Answer:"""
        elif self.model_name == "llama3-8b-8192":
            return """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {query_str} 
            Context: {context_str} 
            Answer:"""
        elif self.model_name == "mixtral-8x7b-32768":
            return """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {query_str} 
            Context: {context_str} 
            Answer:"""
        elif self.model_name == "gemma2-9b-it":
            return """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {query_str} 
            Context: {context_str}
            Answer:"""

    
    
    """TODO
    CUSTOM QA Templates for Each Model
    Start Evaluation Basic.
    """