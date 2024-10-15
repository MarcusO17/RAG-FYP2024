# Core Libraries
import gradio as gr
import shutil
import os
from llama_index.core import Settings,StorageContext,VectorStoreIndex,SimpleDirectoryReader
from dotenv import load_dotenv
from groq import Groq
from groqllm import GroqLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#Chunking
from llama_index.core.node_parser import SentenceSplitter

#VectorStorage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate,Document 
import chromadb


def upload_file(file):
    #UPLOAD_PATH = "./docs"
    #if os.path.exists(UPLOAD_PATH) is not True:
        #os.mkdir(UPLOAD_PATH)
    #shutil.copy(file, UPLOAD_PATH)
    gr.Info('Successful!!')
    load_documents(file)




def load_documents(file):
    pipeline = IngestionPipeline(
        transformations=[
            #Splits chunks to 512 with 50 overlap
            SentenceSplitter(chunk_size=512, chunk_overlap=50), 
        ],
        vector_store=vector_store,
    )

    documents = pipeline.run(file)
    gr.Info('Loading File')

    index = VectorStoreIndex(documents, storage_context=storage_context,similarity_top_k=5) 
    gr.Info('Constructed Index')
    query_engine = index.as_query_engine(streaming=True,similarity_top_k=3)
    gr.Info('Constructed Query Engine')


def respond(message,history):
    response = query_engine.query()
    response.print_response_stream()


load_dotenv()

global llm 
global index
global query_engine

llm = GroqLLM(model_name = "llama3-8b-8192"
            ,client =Groq(api_key=os.getenv("GROQ_API_KEY"))
            ,temperature =0.0
            ,system_prompt = ("""
Instructions:
- You are a helpful assistant
Question: {query_str}
"""))

embed_model = HuggingFaceEmbedding(model_name='Snowflake/snowflake-arctic-embed-m' 
                                   ,trust_remote_code=True
                                   )

db = chromadb.EphemeralClient() #Makes a temporary client which is not on disk
chroma_collection = db.get_or_create_collection("temp")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
Settings.llm = llm
Settings.embed_model = embed_model


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            """
            ## Hello
            """
        )
    with gr.Row():
        # Chat Interface
        with gr.Column(scale=1.2,min_width=100):
            with gr.Tab(label='RAG Chatbot'):
                chatbot = gr.Chatbot(type="messages")
                msg = gr.Textbox()
                msg.submit(respond, [msg, chatbot], [chatbot])
                
            with gr.Tab(label='File Input'):
                upload_button = gr.UploadButton("Click to Upload a File", file_types=['.pdf','.txt','.doc'])
                upload_button.upload(upload_file,upload_button)
                load_btn = gr.Button("Load PDF Documents only")
                load_btn.click(load_documents)


        # Show Nodes
        with gr.Column(scale=0.7,min_width=70):
            gr.Textbox(label="Column 3")
            gr.Button("Button 3")




demo.queue().launch()

