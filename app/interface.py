import gradio as gr
from model import Model


def main():
    model = Model()
        
    image = gr.Image()
    prompt_textbox = gr.Textbox(label='Prompt')
    neg_prompt_textbox = gr.Textbox(label='Negative prompt')
    
    gr.Interface(fn=model.get_picture, 
                inputs= [
                    prompt_textbox,
                    neg_prompt_textbox
                ],
                outputs=image
    ).launch()
    
    

if __name__ == "__main__":
    main()