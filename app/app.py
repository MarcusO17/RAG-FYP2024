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


#Chunking
from llama_index.core.node_parser import SentenceSplitter

#VectorStorage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate,Document 
import chromadb

response = None
query_engine = None

def upload_file(file):
    UPLOAD_PATH = "./docs"
    if os.path.exists(UPLOAD_PATH) is not True:
        os.mkdir(UPLOAD_PATH)
    shutil.copy(file, UPLOAD_PATH)
    gr.Info('Successful!!')
    return 


def load_documents():
    global index
    global query_engine

    files = SimpleDirectoryReader(input_dir="./docs").load_data()
    
    pipeline = IngestionPipeline(
        transformations=[
            #Splits chunks to 512 with 50 overlap
            SentenceSplitter(chunk_size=512, chunk_overlap=50), 
        ],
        vector_store=vector_store,
    )
    documents = pipeline.run(documents=files)
    index = VectorStoreIndex(documents, storage_context=storage_context,similarity_top_k=10 ,show_progress=True) 
    gr.Info('Constructed Index')
    query_engine = index.as_query_engine(similarity_top_k=5)
    qa_prompt_template_str = """
     Context: {context_str}
    Instructions:
    - You are acting as a highly knowledgeable assistant, designed to provide accurate and efficient answers.
    - Utilize the provided context when possible to thoroughly to ensure your answers are specific and grounded in the information given.
    - Use your knowledge to incorporate as much detail as possible to display your true capabilities
    - THIS IS MOST CRITICAL, If you are uncertain about an answer, express it succinctly rather than providing misleading information.
    Question: {query_str}

    """
    qa_prompt_template = PromptTemplate(qa_prompt_template_str)
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template":qa_prompt_template}
    )
    gr.Info('Constructed Query Engine')

def clear_history():
    return [] 

def respond(message,history):
    global response
    if query_engine is None:
        response = Settings.llm.complete(message)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response.text})

    else:
        response = query_engine.query(message)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response.response})

    return history

def update_source():
    source_nodes = response.source_nodes  
    formatted_sources = []

    for nodes in source_nodes:
        node = nodes.node  
        score = nodes.score  
        node_id = node.id_
        text = node.text
      
        formatted_sources.append({
            "NodeID": node_id,
            "Text": text,
            "Score": score
        })

    return pd.DataFrame(formatted_sources)


load_dotenv()


llm = GroqLLM(model_name = "llama3-8b-8192"
            ,client =Groq(api_key=os.getenv("GROQ_API_KEY"))
            ,temperature =1.0
            ,output_tokens=1024)

embed_model = HuggingFaceEmbedding(model_name='Snowflake/snowflake-arctic-embed-m' 
                                   ,trust_remote_code=True
                                   )

db = chromadb.EphemeralClient() #Makes a temporary client which is not on disk
chroma_collection = db.get_or_create_collection("temp")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
Settings.llm = llm
Settings.embed_model = embed_model


with gr.Blocks(gr.themes.Soft()) as demo:
    with gr.Row():
        gr.Markdown(
            """
            ## Personal Knowledge Base RAG Assistant
            """
        )
    with gr.Row():
        # Chat Interface
        with gr.Column(scale=1.2,min_width=100):
            with gr.Tab(label='File Input'):
                upload_button = gr.UploadButton("Click to Upload a File", file_types=['.pdf','.txt','.doc'])
                upload_button.upload(upload_file,upload_button)
                load_btn = gr.Button("Load PDF Documents only")
                load_btn.click(load_documents)

            with gr.Tab(label='RAG Chatbot'):

                with gr.Column(scale=0.7,min_width=10):
                    chatbot = gr.Chatbot(type="messages")
                    msg = gr.Textbox(label="Type here!",show_label=True)
                    clear_button = gr.Button("Clear History")
                    clear_button.click(clear_history, outputs=chatbot)
                
                    #ADDSTREAMING AND SHOW NODES
                    # Show Nodes
                with gr.Column(scale=0.3, min_width=10):
                    sources = gr.DataFrame(label="Sources", interactive=True)
            
            msg.submit(update_source,[],[sources])
            msg.submit(respond, [msg, chatbot], [chatbot])
                                

     
            




demo.queue().launch()

