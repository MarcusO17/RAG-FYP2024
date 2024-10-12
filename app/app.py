# Core Libraries
import gradio as gr

def reply(prompt):
    return "Hello!"

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            """
            ## Hello
            """
        )
    with gr.Row():

        # File Handling
        with gr.Column(scale=0.3,min_width=50):
            with gr.Row():
                upload_button = gr.UploadButton("Click to Upload a File", file_types=["text"])



        # Chat Interface
        with gr.Column(scale=1,min_width=100):
            gr.Textbox(label="Column 2")
            gr.Button("Button 2")
        

        # Show Nodes
        with gr.Column(scale=1,min_width=100):
            gr.Textbox(label="Column 3")
            gr.Button("Button 3")




demo.launch()