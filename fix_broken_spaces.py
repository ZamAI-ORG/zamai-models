#!/usr/bin/env python3
"""
Fix broken HuggingFace spaces and train missing models
"""

import os
from huggingface_hub import HfApi, upload_file
import tempfile

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def fix_hf_inference_space():
    """Fix the HF-Inference space that's missing app.py"""
    token = read_hf_token()
    space_id = "tasal9/HF-Inference"
    
    print(f"🔧 Fixing space: {space_id}")
    
    # Create a proper inference app
    app_content = '''import gradio as gr
from huggingface_hub import InferenceClient
import os

# Initialize inference clients for your models
models = {
    "Pashto Chat": "tasal9/ZamAI-LIama3-Pashto",
    "Pashto BLOOM": "tasal9/pashto-base-bloom", 
    "Mistral Pashto": "tasal9/ZamAI-Mistral-7B-Pashto",
    "Phi-3 Business": "tasal9/ZamAI-Phi-3-Mini-Pashto"
}

def generate_text(model_choice, prompt, max_tokens=150, temperature=0.7):
    """Generate text using selected model"""
    try:
        model_id = models[model_choice]
        client = InferenceClient(model_id)
        
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def transcribe_audio(audio):
    """Transcribe audio using Whisper model"""
    try:
        client = InferenceClient("tasal9/ZamAI-Whisper-v3-Pashto")
        result = client.automatic_speech_recognition(audio)
        return result.get('text', 'No transcription available')
    except Exception as e:
        return f"Error: {str(e)}"

def get_embeddings(text):
    """Get embeddings for text"""
    try:
        client = InferenceClient("tasal9/Multilingual-ZamAI-Embeddings")
        embeddings = client.feature_extraction(text)
        return f"Embedding vector (first 10 dims): {embeddings[:10]}..."
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="ZamAI HuggingFace Inference Hub") as demo:
    gr.Markdown("# 🤗 ZamAI HuggingFace Inference Hub")
    gr.Markdown("Test all your ZamAI models in one place!")
    
    with gr.Tabs():
        # Text Generation Tab
        with gr.TabItem("💬 Text Generation"):
            with gr.Row():
                with gr.Column():
                    model_choice = gr.Dropdown(
                        choices=list(models.keys()),
                        value="Pashto Chat",
                        label="Select Model"
                    )
                    prompt = gr.Textbox(
                        label="Enter your prompt",
                        placeholder="د افغانستان په اړه لیکل...",
                        lines=3
                    )
                    max_tokens = gr.Slider(10, 300, value=150, label="Max Tokens")
                    temperature = gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
                    generate_btn = gr.Button("🚀 Generate", variant="primary")
                
                with gr.Column():
                    text_output = gr.Textbox(label="Generated Text", lines=8)
            
            generate_btn.click(
                generate_text,
                inputs=[model_choice, prompt, max_tokens, temperature],
                outputs=text_output
            )
        
        # Speech Recognition Tab
        with gr.TabItem("🎤 Speech Recognition"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Record or Upload Audio"
                    )
                    transcribe_btn = gr.Button("📝 Transcribe", variant="primary")
                
                with gr.Column():
                    transcription = gr.Textbox(label="Transcription", lines=5)
            
            transcribe_btn.click(transcribe_audio, inputs=audio_input, outputs=transcription)
        
        # Embeddings Tab
        with gr.TabItem("🌐 Embeddings"):
            with gr.Row():
                with gr.Column():
                    embed_text = gr.Textbox(
                        label="Enter text for embeddings",
                        placeholder="Any text in any language...",
                        lines=3
                    )
                    embed_btn = gr.Button("🔍 Get Embeddings", variant="primary")
                
                with gr.Column():
                    embed_output = gr.Textbox(label="Embeddings", lines=5)
            
            embed_btn.click(get_embeddings, inputs=embed_text, outputs=embed_output)

if __name__ == "__main__":
    demo.launch()
'''

    requirements_content = '''gradio>=4.8.0
huggingface_hub>=0.19.0
'''

    try:
        # Upload app.py
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(app_content)
            app_path = f.name
        
        upload_file(
            path_or_fileobj=app_path,
            path_in_repo="app.py",
            repo_id=space_id,
            repo_type="space",
            token=token,
            commit_message="Add missing app.py for inference hub"
        )
        
        # Upload requirements.txt
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(requirements_content)
            req_path = f.name
        
        upload_file(
            path_or_fileobj=req_path,
            path_in_repo="requirements.txt",
            repo_id=space_id,
            repo_type="space",
            token=token,
            commit_message="Add requirements.txt"
        )
        
        # Cleanup
        os.unlink(app_path)
        os.unlink(req_path)
        
        print(f"   ✅ Fixed {space_id}")
        
    except Exception as e:
        print(f"   ❌ Error fixing {space_id}: {e}")

def fix_multimodel_playground():
    """Fix the ZamAI-Pashto-Multimodel-AI-Playground space"""
    token = read_hf_token()
    space_id = "tasal9/ZamAI-Pashto-Multimodel-AI-Playground"
    
    print(f"🔧 Fixing space: {space_id}")
    
    # Create multimodel playground app
    app_content = '''import gradio as gr
from huggingface_hub import InferenceClient
import numpy as np

# Your ZamAI models
MODELS = {
    "text_generation": {
        "LLaMA3 Pashto": "tasal9/ZamAI-LIama3-Pashto",
        "BLOOM Pashto": "tasal9/pashto-base-bloom",
        "Mistral Pashto": "tasal9/ZamAI-Mistral-7B-Pashto",
        "Phi-3 Business": "tasal9/ZamAI-Phi-3-Mini-Pashto"
    },
    "speech": "tasal9/ZamAI-Whisper-v3-Pashto",
    "embeddings": "tasal9/Multilingual-ZamAI-Embeddings"
}

def compare_models(prompt, max_tokens=100):
    """Compare outputs from different text generation models"""
    results = {}
    
    for model_name, model_id in MODELS["text_generation"].items():
        try:
            client = InferenceClient(model_id)
            response = client.text_generation(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True
            )
            results[model_name] = response
        except Exception as e:
            results[model_name] = f"Error: {str(e)}"
    
    return results["LLaMA3 Pashto"], results["BLOOM Pashto"], results["Mistral Pashto"], results["Phi-3 Business"]

def process_speech(audio):
    """Process speech input"""
    try:
        client = InferenceClient(MODELS["speech"])
        result = client.automatic_speech_recognition(audio)
        return result.get('text', 'No transcription')
    except Exception as e:
        return f"Error: {str(e)}"

def compare_embeddings(text1, text2):
    """Compare embeddings between two texts"""
    try:
        client = InferenceClient(MODELS["embeddings"])
        
        emb1 = client.feature_extraction(text1)
        emb2 = client.feature_extraction(text2)
        
        # Calculate similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return f"Similarity: {similarity:.4f}", f"Text 1 embedding: {emb1[:5]}...", f"Text 2 embedding: {emb2[:5]}..."
        
    except Exception as e:
        return f"Error: {str(e)}", "", ""

# Create the interface
with gr.Blocks(title="ZamAI Multimodel Playground") as demo:
    gr.Markdown("# 🇦🇫 ZamAI Multimodel AI Playground")
    gr.Markdown("Compare and test all your ZamAI models in one interface!")
    
    with gr.Tabs():
        # Model Comparison Tab
        with gr.TabItem("⚖️ Model Comparison"):
            gr.Markdown("### Compare outputs from different text generation models")
            
            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Enter prompt for comparison",
                    placeholder="د افغانستان په اړه لیکل...",
                    lines=3
                )
                max_tokens = gr.Slider(50, 200, value=100, label="Max Tokens")
            
            compare_btn = gr.Button("🚀 Compare Models", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    llama_output = gr.Textbox(label="LLaMA3 Pashto", lines=5)
                    mistral_output = gr.Textbox(label="Mistral Pashto", lines=5)
                with gr.Column():
                    bloom_output = gr.Textbox(label="BLOOM Pashto", lines=5)
                    phi3_output = gr.Textbox(label="Phi-3 Business", lines=5)
            
            compare_btn.click(
                compare_models,
                inputs=[prompt_input, max_tokens],
                outputs=[llama_output, bloom_output, mistral_output, phi3_output]
            )
        
        # Speech Processing Tab
        with gr.TabItem("🎤 Speech Processing"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Speak in Pashto"
                    )
                    speech_btn = gr.Button("🎯 Process Speech", variant="primary")
                
                with gr.Column():
                    speech_output = gr.Textbox(label="Transcription", lines=5)
            
            speech_btn.click(process_speech, inputs=audio_input, outputs=speech_output)
        
        # Embeddings Comparison Tab
        with gr.TabItem("🌐 Embeddings Comparison"):
            with gr.Row():
                with gr.Column():
                    text1 = gr.Textbox(label="First Text", placeholder="Enter first text...")
                    text2 = gr.Textbox(label="Second Text", placeholder="Enter second text...")
                    embed_btn = gr.Button("🔍 Compare Embeddings", variant="primary")
                
                with gr.Column():
                    similarity_output = gr.Textbox(label="Similarity Score", lines=2)
                    emb1_output = gr.Textbox(label="Embedding 1", lines=2)
                    emb2_output = gr.Textbox(label="Embedding 2", lines=2)
            
            embed_btn.click(
                compare_embeddings,
                inputs=[text1, text2],
                outputs=[similarity_output, emb1_output, emb2_output]
            )

if __name__ == "__main__":
    demo.launch()
'''

    requirements_content = '''gradio>=4.8.0
huggingface_hub>=0.19.0
numpy>=1.21.0
'''

    try:
        # Upload app.py
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(app_content)
            app_path = f.name
        
        upload_file(
            path_or_fileobj=app_path,
            path_in_repo="app.py",
            repo_id=space_id,
            repo_type="space",
            token=token,
            commit_message="Add missing app.py for multimodel playground"
        )
        
        # Upload requirements.txt
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(requirements_content)
            req_path = f.name
        
        upload_file(
            path_or_fileobj=req_path,
            path_in_repo="requirements.txt",
            repo_id=space_id,
            repo_type="space",
            token=token,
            commit_message="Add requirements.txt"
        )
        
        # Cleanup
        os.unlink(app_path)
        os.unlink(req_path)
        
        print(f"   ✅ Fixed {space_id}")
        
    except Exception as e:
        print(f"   ❌ Error fixing {space_id}: {e}")

def fix_education_bot_space():
    """Add missing requirements.txt to education bot"""
    token = read_hf_token()
    space_id = "tasal9/ZamAI-Pashto-Education-Bot-Demo"
    
    print(f"🔧 Adding requirements.txt to: {space_id}")
    
    requirements_content = '''gradio>=4.8.0
transformers>=4.30.0
torch>=2.0.0
accelerate>=0.20.0
'''

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(requirements_content)
            req_path = f.name
        
        upload_file(
            path_or_fileobj=req_path,
            path_in_repo="requirements.txt",
            repo_id=space_id,
            repo_type="space",
            token=token,
            commit_message="Add missing requirements.txt"
        )
        
        os.unlink(req_path)
        print(f"   ✅ Fixed {space_id}")
        
    except Exception as e:
        print(f"   ❌ Error fixing {space_id}: {e}")

def main():
    print("🔧 FIXING BROKEN HUGGINGFACE SPACES")
    print("=" * 50)
    
    # Fix the broken spaces
    fix_hf_inference_space()
    fix_multimodel_playground()
    fix_education_bot_space()
    
    print("\\n✅ Space fixes completed!")
    print("\\n🎯 Next steps:")
    print("1. Check the fixed spaces are working")
    print("2. Train missing models using your training space")
    print("3. Upload any local models that aren't on the hub")

if __name__ == "__main__":
    main()
