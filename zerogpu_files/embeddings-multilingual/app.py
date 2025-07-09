import gradio as gr
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spaces

MODEL_ID = "tasal9/Multilingual-ZamAI-Embeddings"

@spaces.GPU
def load_embeddings_model():
    """Load embeddings model with ZeroGPU support"""
    model = SentenceTransformer(MODEL_ID)
    return model

@spaces.GPU
def get_embeddings(text_input):
    """Get embeddings for input text"""
    try:
        model = load_embeddings_model()
        embeddings = model.encode([text_input])
        return embeddings[0].tolist()[:10]  # Return first 10 dimensions
        
    except Exception as e:
        return f"Error: {e}"

@spaces.GPU
def calculate_similarity(text1, text2):
    """Calculate semantic similarity between two texts"""
    try:
        model = load_embeddings_model()
        
        embeddings = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return f"Similarity Score: {similarity:.4f}"
        
    except Exception as e:
        return f"Error: {e}"

# Gradio Interface
with gr.Blocks(title="ZamAI Embeddings Training", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 ZamAI Embeddings Training Space")
    gr.Markdown(f"**Model:** {MODEL_ID}")
    gr.Markdown("This space allows you to test and fine-tune your multilingual embeddings model.")
    
    with gr.Tab("🔍 Test Embeddings"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Enter text (Pashto/English/etc.)",
                    placeholder="زموږ د ژبې موډل ازمویاست",
                    lines=3
                )
                embed_btn = gr.Button("🔮 Get Embeddings", variant="primary")
            
            with gr.Column():
                embeddings_output = gr.Textbox(
                    label="Embeddings (first 10 dimensions)",
                    lines=8,
                    interactive=False
                )
        
        embed_btn.click(
            get_embeddings,
            inputs=text_input,
            outputs=embeddings_output
        )
    
    with gr.Tab("📊 Similarity"):
        with gr.Row():
            with gr.Column():
                text1_input = gr.Textbox(
                    label="Text 1",
                    placeholder="سلام ورور",
                    lines=2
                )
                text2_input = gr.Textbox(
                    label="Text 2",
                    placeholder="Hello brother", 
                    lines=2
                )
                similarity_btn = gr.Button("📊 Calculate Similarity", variant="primary")
            
            with gr.Column():
                similarity_output = gr.Textbox(
                    label="Similarity Result",
                    lines=4,
                    interactive=False
                )
        
        similarity_btn.click(
            calculate_similarity,
            inputs=[text1_input, text2_input],
            outputs=similarity_output
        )

if __name__ == "__main__":
    demo.launch()
