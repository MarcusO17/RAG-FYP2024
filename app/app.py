# Core Libraries
import gradio as gr
import shutil
import os
from llama_index.core import Settings,StorageContext,VectorStoreIndex,SimpleDirectoryReader
from dotenv import load_dotenv
from groq import Groq
from groqllm import GroqLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pandas as pd
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from sklearn.decomposition import PCA
import plotly.express as px

#PERFORMANCE TESTING
from transformers import AutoTokenizer
from llama_index.core.callbacks import TokenCountingHandler

#Chunking
from llama_index.core.node_parser import  SemanticSplitterNodeParser

#VectorStorage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate 
import chromadb



response = None
query_engine = None

def upload_file(file):
    UPLOAD_PATH = "./docs"
    if os.path.exists(UPLOAD_PATH) is not True:
        os.mkdir(UPLOAD_PATH)
    shutil.copy(file, UPLOAD_PATH)
    gr.Info('Successful!!')
    return get_file_list()


def load_documents():
    global index
    global query_engine
    query_engine = None

    print('Loading Documents')
    files = SimpleDirectoryReader(input_dir="./docs").load_data()

    pipeline = IngestionPipeline(
        transformations=[
            SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model, show_progress=True
        )
        ],
        vector_store=vector_store,
    )
    documents = pipeline.run(documents=files)
    
    index = VectorStoreIndex(documents, storage_context=storage_context,similarity_top_k=10 ,node_postprocessors=[reranker_model],show_progress=True)
    gr.Info('Constructed Index')

    query_engine = index.as_chat_engine(similarity_top_k=5,streaming=True,chat_mode="context")
    gr.Info('Constructed Query Engine')

    fig = get_embedding_space()
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    
    return fig

def clear_history():
    return []

def get_file_list():
    return os.listdir('./docs')

def respond(message,history):
    global response
    if query_engine is None:
        generator = llm.stream_complete(message)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content":""})
        for response in generator:
            history[-1]['content'] += response.delta #Take last message and add
            yield history

    else:
        response = query_engine.stream_chat(message)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        for text in response.response_gen:
            history[-1]['content'] += text #Take last message and add
            yield history


def update_source():
    try:   
        source_nodes = response.source_nodes
    
        formatted_sources = []

        for nodes in source_nodes:
            node = nodes.node
            score = nodes.score
            doc_name = nodes.file_name
            text = node.text

            formatted_sources.append({
                "Document": doc_name,
                "Text": text,
                "Score": score
            })

        return pd.DataFrame(formatted_sources)
    except:
        return pd.DataFrame(columns=["NodeID","Text","Score"])

def get_embedding_space():
    pca = PCA(n_components=3)
    embeddings = chroma_collection.get(include=['embeddings'])

    if embeddings and len(embeddings['embeddings']) > 0:
        emb_transform = pca.fit_transform(embeddings['embeddings'])
        fig = px.scatter_3d(
            x=emb_transform[:, 0],
            y=emb_transform[:, 1],
            z=emb_transform[:, 2],
        )
        return fig
    else:
        return px.scatter_3d(title="Knowledge Base Empty")

load_dotenv()

llm = GroqLLM(model_name = "llama-3.1-70b-versatile"
            ,client =Groq(api_key=os.environ.get('GROQ_API_KEY'))
            ,temperature =1.0
            ,output_tokens=1024
            ,system_prompt=""" 
                        You are a wise and knowledgeable mentor who excels at guiding others to understanding. Think of yourself as a trusted advisor who has deep expertise but speaks in an approachable way.
            Context: {context_str}
            As you help answer this question, please:

            Take a moment to absorb the context I've provided. What are the key insights that relate to what's being asked? Consider both the explicit information and any relevant implications.
            Draw from both this context and your broader knowledge, but be clear about which is which. You might say things like "From what we can see in the provided information..." or "While the context doesn't directly address this, my knowledge suggests..."
            Structure your guidance naturally:

            Start with a clear, direct response
            Build understanding gradually, like explaining to a curious learner
            Share deeper insights when relevant, but avoid overwhelming
            If you're unsure about something, simply say so - it's better to be honest than misleading

            When you respond, try to:

            Connect ideas in an intuitive way
            Use analogies or examples when they help clarify
            Point out particularly interesting or important aspects
            Acknowledge any limitations or uncertainties in your understanding

            Question: {query_str}
            Remember, your role is to guide and illuminate, not just to inform. Share your knowledge in a way that helps build understanding while maintaining accuracy and trustworthiness.
            Shape your response as a natural dialogue, but ensure it includes:

            A clear initial answer
            Supporting explanations that build in complexity
            Relevant insights from the context
            Honest acknowledgment of any uncertainties
            Connections to broader understanding when helpful
            """)

embed_model = HuggingFaceEmbedding(model_name='Snowflake/snowflake-arctic-embed-m'
                                   ,trust_remote_code=True,
                                    device="cuda")

db = chromadb.EphemeralClient() #Makes a temporary client which is not on disk
chroma_collection = db.get_or_create_collection("temp")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
Settings.llm = llm
Settings.embed_model = embed_model
reranker_model = FlagEmbeddingReranker(model="mixedbread-ai/mxbai-embed-large-v1", top_n=5)

with gr.Blocks(gr.themes.Soft()) as demo:
    with gr.Row():
        gr.Markdown(
            """
            ## Personal Knowledge Base RAG Assistant
            """
        )
    with gr.Row():
        # Chat Interface
        with gr.Column(scale=3):
            with gr.Tab():
                file_list = gr.CheckboxGroup(choices=os.listdir('./docs'),label="Files", info="Choose your files to insert!",interactive=True)
                upload_button = gr.UploadButton("Click to Upload a File", file_types=['.pdf','.txt','.doc'])
                upload_button.upload(upload_file,upload_button,[file_list])
                load_btn = gr.Button("Load PDF Documents only")
            with gr.Tab():
                chatbot = gr.Chatbot(type="messages")
                msg = gr.Textbox(label="Type here!",show_label=True)
                clear_button = gr.Button("Clear History")
                clear_button.click(clear_history, outputs=chatbot)
        # Show Nodes
        with gr.Column(scale=2):
            with gr.Accordion("See Details"):
                sources = gr.DataFrame(label="Sources", interactive=True)
            with gr.Row():
                embed_plot = gr.Plot(label="Embedding Plot")

                
                
  
        load_btn.click(load_documents,outputs=[embed_plot])
        msg.submit(update_source,[],[sources])
        msg.submit(respond, [msg, chatbot], [chatbot])

demo.queue().launch(share=True)
     

