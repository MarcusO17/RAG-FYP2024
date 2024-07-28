import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
class Components:
    def __init__(self,embed,reranker):
        self.embed = embed
        self.reranker = reranker

    def get_embedding_model(self):
        return HuggingFaceEmbedding(model_name = self.embed)

    def get_reranker(self,reranker):
        return FlagEmbeddingReranker(model=reranker, top_n=5)
    
    
        