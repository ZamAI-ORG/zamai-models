#!/usr/bin/env python3
"""
Create all ZamAI Hugging Face Spaces
"""

import os
from huggingface_hub import HfApi, create_repo, upload_file
import tempfile
import time

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def create_space_files(space_config):
    """Create the files for a space"""
    
    # README.md
    readme = f"""---
title: {space_config['title']}
emoji: {space_config['emoji']}
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: {space_config.get('hardware', 'zero-a10g')}
models:
- {space_config['model_id']}
---

# {space_config['emoji']} {space_config['title']}

{space_config['description']}

## Features
- ✅ Real-time inference
- ✅ User-friendly interface  
- ✅ Optimized for performance
- ✅ Multilingual support (Pashto/English)

## Model
This space uses the [{space_config['model_id']}](https://huggingface.co/{space_config['model_id']}) model.
"""

    # app.py - Text Generation
    if space_config['task'] == 'text-generation':
        app = f'''import gradio as gr
from transformers import pipeline
import torch
import spaces

# Load model
model_id = "{space_config['model_id']}"
generator = None

@spaces.GPU
def load_model():
    global generator
    try:
        generator = pipeline(
            "text-generation", 
            model=model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        return f"✅ Model loaded successfully: {{model_id}}"
    except Exception as e:
        return f"❌ Error loading model: {{str(e)[:200]}}"

@spaces.GPU
def generate_text(prompt, max_length=150, temperature=0.7):
    if generator is None:
        return "Please load the model first using the 'Load Model' button."
    
    if not prompt.strip():
        return "Please enter a prompt."
    
    try:
        result = generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
            truncation=True
        )
        return result[0]['generated_text']
    except Exception as e:
        return f"Error generating text: {{str(e)[:200]}}"

# Gradio Interface
with gr.Blocks(title="{space_config['title']}") as demo:
    gr.Markdown("# {space_config['emoji']} {space_config['title']}")
    gr.Markdown("{space_config['description']}")
    
    with gr.Row():
        load_btn = gr.Button("🔄 Load Model", variant="secondary")
        status = gr.Textbox(label="Status", value="Click 'Load Model' to start")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Enter your prompt",
                placeholder="د افغانستان په اړه لیکل...",
                lines=3
            )
            max_length = gr.Slider(10, 300, value=150, label="Max Length")
            temperature = gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
            generate_btn = gr.Button("🚀 Generate", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Generated Text",
                lines=8
            )
    
    load_btn.click(load_model, outputs=status)
    generate_btn.click(
        generate_text,
        inputs=[prompt, max_length, temperature],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
'''

    # app.py - Speech Recognition
    elif space_config['task'] == 'automatic-speech-recognition':
        app = f'''import gradio as gr
from transformers import pipeline
import torch
import spaces

# Primary model with fallback
model_id = "{space_config['model_id']}"
fallback_model = "openai/whisper-base"
transcriber = None

@spaces.GPU
def load_model():
    global transcriber
    try:
        # Try primary model first
        transcriber = pipeline("automatic-speech-recognition", model=model_id)
        return f"✅ Model loaded: {{model_id}}"
    except Exception as e:
        try:
            # Fallback to official Whisper model
            transcriber = pipeline("automatic-speech-recognition", model=fallback_model)
            return f"✅ Fallback model loaded: {{fallback_model}} (Primary model issue: {{str(e)[:100]}})"
        except Exception as e2:
            return f"❌ Error loading models: {{str(e2)}}"

@spaces.GPU
def transcribe_audio(audio):
    if transcriber is None:
        return "Please load the model first."
    
    if audio is None:
        return "Please provide audio input."
    
    try:
        result = transcriber(audio)
        return result["text"]
    except Exception as e:
        return f"Error: {{str(e)}}"

with gr.Blocks(title="{space_config['title']}") as demo:
    gr.Markdown("# {space_config['emoji']} {space_config['title']}")
    gr.Markdown("{space_config['description']}")
    
    with gr.Row():
        load_btn = gr.Button("🔄 Load Model")
        status = gr.Textbox(label="Status", value="Click to load model")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎤 Record or Upload Audio"
            )
            transcribe_btn = gr.Button("📝 Transcribe", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(label="Transcription", lines=5)
    
    load_btn.click(load_model, outputs=status)
    transcribe_btn.click(transcribe_audio, inputs=audio_input, outputs=output)

if __name__ == "__main__":
    demo.launch()
'''

    # app.py - Embeddings
    elif space_config['task'] == 'feature-extraction':
        app = f'''import gradio as gr
from sentence_transformers import SentenceTransformer
import numpy as np
import spaces

model_id = "{space_config['model_id']}"
model = None

@spaces.GPU
def load_model():
    global model
    try:
        model = SentenceTransformer(model_id)
        return "✅ Embeddings model loaded!"
    except Exception as e:
        return f"❌ Error: {{str(e)}}"

@spaces.GPU
def get_embeddings(text1, text2=""):
    if model is None:
        return "Please load model first.", 0.0
    
    try:
        texts = [text1]
        if text2.strip():
            texts.append(text2)
        
        embeddings = model.encode(texts)
        
        if len(texts) == 2:
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return f"Similarity: {{similarity:.4f}}\\n\\nEmbedding 1: {{embeddings[0][:10]}}...\\nEmbedding 2: {{embeddings[1][:10]}}...", float(similarity)
        else:
            return f"Embedding: {{embeddings[0][:10]}}...", 0.0
    except Exception as e:
        return f"Error: {{str(e)}}", 0.0

with gr.Blocks(title="{space_config['title']}") as demo:
    gr.Markdown("# {space_config['emoji']} {space_config['title']}")
    gr.Markdown("{space_config['description']}")
    
    with gr.Row():
        load_btn = gr.Button("🔄 Load Model")
        status = gr.Textbox(label="Status", value="Click to load model")
    
    with gr.Row():
        with gr.Column():
            text1 = gr.Textbox(label="Text 1", placeholder="Enter first text...")
            text2 = gr.Textbox(label="Text 2 (optional)", placeholder="Enter second text for similarity...")
            process_btn = gr.Button("🔍 Process", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(label="Results", lines=5)
            similarity = gr.Number(label="Similarity Score", precision=4)
    
    load_btn.click(load_model, outputs=status)
    process_btn.click(get_embeddings, inputs=[text1, text2], outputs=[output, similarity])

if __name__ == "__main__":
    demo.launch()
'''
    else:
        # Default fallback app
        app = f'''import gradio as gr
import spaces

@spaces.GPU
def placeholder_function(input_text):
    return f"Model {space_config['model_id']} is being set up. Please try again later."

demo = gr.Interface(
    fn=placeholder_function,
    inputs="text",
    outputs="text",
    title="{space_config['title']}"
)

if __name__ == "__main__":
    demo.launch()
'''

    # requirements.txt
    requirements = """gradio>=4.8.0
transformers>=4.30.0
torch>=2.0.0
sentence-transformers>=2.2.0
numpy>=1.21.0
accelerate>=0.20.0
spaces
"""

    return readme, app, requirements

def create_single_space(space_config, token):
    """Create a single HuggingFace space"""
    
    space_name = f"tasal9/{space_config['name']}"
    print(f"\\n🚀 Creating: {space_name}")
    
    try:
        # Create repository
        create_repo(
            repo_id=space_name,
            repo_type="space",
            space_sdk="gradio",
            token=token,
            exist_ok=True
        )
        print("   ✅ Repository created")
        
        # Get file contents
        readme, app, requirements = create_space_files(space_config)
        
        # Create temp files and upload
        files_to_upload = [
            ("README.md", readme),
            ("app.py", app),
            ("requirements.txt", requirements)
        ]
        
        for filename, content in files_to_upload:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                upload_file(
                    path_or_fileobj=temp_path,
                    path_in_repo=filename,
                    repo_id=space_name,
                    repo_type="space",
                    token=token,
                    commit_message=f"Update {filename}"
                )
                print(f"   ✅ {filename} uploaded")
            except Exception as e:
                print(f"   ⚠️  {filename} upload issue: {e}")
            finally:
                os.unlink(temp_path)
        
        print(f"   🌐 https://huggingface.co/spaces/{space_name}")
        return space_name
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    print("🚀 Creating ZamAI Hugging Face Spaces")
    print("=" * 50)
    
    token = read_hf_token()
    
    # Define all spaces
    spaces = [
        {
            "name": "zamai-pashto-chat",
            "title": "ZamAI Pashto Chat",
            "emoji": "💬",
            "description": "Advanced conversational AI for Pashto language with cultural context understanding.",
            "model_id": "tasal9/ZamAI-LIama3-Pashto",
            "task": "text-generation"
        },
        {
            "name": "zamai-bloom-pashto", 
            "title": "ZamAI BLOOM Pashto",
            "emoji": "🌸",
            "description": "High-quality Pashto text generation using BLOOM architecture.",
            "model_id": "tasal9/pashto-base-bloom",
            "task": "text-generation"
        },
        {
            "name": "zamai-mistral-pashto",
            "title": "ZamAI Mistral Pashto", 
            "emoji": "🚀",
            "description": "State-of-the-art Pashto text generation using Mistral-7B.",
            "model_id": "tasal9/ZamAI-Mistral-7B-Pashto",
            "task": "text-generation"
        },
        {
            "name": "zamai-phi3-business",
            "title": "ZamAI Phi-3 Business",
            "emoji": "💼", 
            "description": "Business automation and document processing in Pashto.",
            "model_id": "tasal9/ZamAI-Phi-3-Mini-Pashto",
            "task": "text-generation"
        },
        {
            "name": "zamai-whisper-speech",
            "title": "ZamAI Whisper Speech", 
            "emoji": "🎤",
            "description": "Convert Pashto speech to text with high accuracy.",
            "model_id": "tasal9/ZamAI-Whisper-v3-Pashto",
            "task": "automatic-speech-recognition",
            "hardware": "zero-a10g"
        },
        {
            "name": "zamai-embeddings",
            "title": "ZamAI Multilingual Embeddings",
            "emoji": "🌐",
            "description": "Generate embeddings and calculate similarity for multilingual text.",
            "model_id": "tasal9/Multilingual-ZamAI-Embeddings", 
            "task": "feature-extraction"
        }
    ]
    
    created_spaces = []
    
    for space_config in spaces:
        space_name = create_single_space(space_config, token)
        if space_name:
            created_spaces.append(space_name)
        time.sleep(2)  # Small delay between creations
    
    print("\\n" + "=" * 50)
    print(f"✅ Created {len(created_spaces)} spaces:")
    for space in created_spaces:
        print(f"   🌐 https://huggingface.co/spaces/{space}")
    
    print("\\n🎯 Next steps:")
    print("1. Visit each space to test functionality")
    print("2. Customize interfaces as needed") 
    print("3. Share with the community!")

if __name__ == "__main__":
    main()
