# Core Libraries
import gradio as gr
import shutil
import os

def upload_file(file):
    UPLOAD_PATH = "./docs"
    if os.path.exists(UPLOAD_PATH) is not True:
        os.mkdir(UPLOAD_PATH)
    shutil.copy(file, UPLOAD_PATH)
    gr.Info('Successful!!')

def list_files():
    try:
        files = os.listdir("./docs")
        if files:
            # Prepare a list of file names and sizes for table display
            file_names = []
            for file in files:
                file_names.append(file)
            return file_names
        else:
            return [["No files found", ""]]
    except Exception as e:
        return [[f"Error: {str(e)}", ""]]

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
                clear = gr.ClearButton([msg, chatbot])
            with gr.Tab(label='File Input'):
                upload_button = gr.UploadButton("Click to Upload a File", file_types=['.pdf','.txt','.doc'])
                upload_button.upload(upload_file,upload_button)
    

        # Show Nodes
        with gr.Column(scale=0.7,min_width=70):
            gr.Textbox(label="Column 3")
            gr.Button("Button 3")




demo.launch()