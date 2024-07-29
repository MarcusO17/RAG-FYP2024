import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os
import chromadb
class Components:
    load_dotenv()

    def __init__(self,embed,reranker):
        self.embed = embed
        self.reranker = reranker

    def get_embedding_model(self):
        embedding_model = HuggingFaceEmbedding(model_name = self.embed)
        print("Embedding model loaded!")
        return embedding_model

    def get_reranker(self):
        reranker_model = FlagEmbeddingReranker(model=self.reranker, top_n=10)
        print("Reranker model loaded!")
        return reranker_model
    
    def get_db(self):
        db = chromadb.EphemeralClient()
        chroma_collection = db.get_or_create_collection("temp")
        return chroma_collection
    
    def get_groq_llm(self,model_name):
        return Groq(model_name,api_key=os.getenv("GROQ_API_KEY"))
    
    """TODO
    CUSTOM QA Templates for Each Model
    Start Evaluation Basic.
    """