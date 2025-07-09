#!/usr/bin/env python3
"""
Create and deploy Hugging Face Spaces for ZamAI models
"""

import os
import shutil
from huggingface_hub import HfApi, create_repo, upload_folder
import json

def read_hf_token():
    """Read HF token from file"""
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def create_space_config(space_name, title, emoji, description, model_id=None):
    """Create README.md config for a space"""
    hardware = "zero-a10g" if "training" in space_name.lower() else "cpu-basic"
    
    config = f"""---
title: {title}
emoji: {emoji}
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: {hardware}
"""
    
    if model_id:
        config += f"models:\n- {model_id}\n"
    
    config += f"""---

# {emoji} {title}

{description}

## Features
- ✅ Real-time inference
- ✅ User-friendly interface
- ✅ Optimized for performance
- ✅ Multilingual support (Pashto/English)

## Usage
Simply type your input and get AI-powered results!

## Model Information
"""
    
    if model_id:
        config += f"This space uses the [{model_id}](https://huggingface.co/{model_id}) model.\n"
    
    return config

def create_basic_app(model_id, task_type="text-generation"):
    """Create a basic Gradio app for a model"""
    
    if task_type == "text-generation":
        app_code = f'''import gradio as gr
from transformers import pipeline
import torch

# Load the model
model_id = "{model_id}"
try:
    generator = pipeline("text-generation", model=model_id, torch_dtype=torch.float16, device_map="auto")
except Exception as e:
    print(f"Error loading model: {{e}}")
    generator = None

def generate_text(prompt, max_length=100, temperature=0.7):
    if generator is None:
        return "Model not available. Please try again later."
    
    try:
        result = generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        return result[0]['generated_text']
    except Exception as e:
        return f"Error generating text: {{str(e)}}"

# Create Gradio interface
with gr.Blocks(title="ZamAI Pashto Text Generation") as demo:
    gr.Markdown("# 🇦🇫 ZamAI Pashto Text Generation")
    gr.Markdown("Generate text in Pashto using advanced AI models")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Enter your prompt (Pashto or English)",
                placeholder="د افغانستان په اړه لیکل..."
            )
            max_length = gr.Slider(
                minimum=10,
                maximum=500,
                value=100,
                label="Maximum Length"
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                label="Temperature"
            )
            generate_btn = gr.Button("Generate Text", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Generated Text",
                lines=10
            )
    
    generate_btn.click(
        generate_text,
        inputs=[prompt, max_length, temperature],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
'''
    
    elif task_type == "automatic-speech-recognition":
        app_code = f'''import gradio as gr
from transformers import pipeline
import torch

# Load the model
model_id = "{model_id}"
try:
    transcriber = pipeline("automatic-speech-recognition", model=model_id)
except Exception as e:
    print(f"Error loading model: {{e}}")
    transcriber = None

def transcribe_audio(audio):
    if transcriber is None:
        return "Model not available. Please try again later."
    
    try:
        result = transcriber(audio)
        return result["text"]
    except Exception as e:
        return f"Error transcribing audio: {{str(e)}}"

# Create Gradio interface
with gr.Blocks(title="ZamAI Pashto Speech Recognition") as demo:
    gr.Markdown("# 🎤 ZamAI Pashto Speech Recognition")
    gr.Markdown("Convert Pashto speech to text using AI")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or Upload Audio"
            )
            transcribe_btn = gr.Button("Transcribe", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Transcribed Text",
                lines=5
            )
    
    transcribe_btn.click(
        transcribe_audio,
        inputs=audio_input,
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
'''
    
    elif task_type == "feature-extraction":
        app_code = f'''import gradio as gr
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Load the model
model_id = "{model_id}"
try:
    model = SentenceTransformer(model_id)
except Exception as e:
    print(f"Error loading model: {{e}}")
    model = None

def get_embeddings(text1, text2=""):
    if model is None:
        return "Model not available.", 0.0
    
    try:
        texts = [text1]
        if text2.strip():
            texts.append(text2)
        
        embeddings = model.encode(texts)
        
        if len(texts) == 2:
            # Calculate similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return f"Embedding 1: {{embeddings[0][:10]}}...\\nEmbedding 2: {{embeddings[1][:10]}}...", float(similarity)
        else:
            return f"Embedding: {{embeddings[0][:10]}}...", 0.0
    
    except Exception as e:
        return f"Error: {{str(e)}}", 0.0

# Create Gradio interface
with gr.Blocks(title="ZamAI Multilingual Embeddings") as demo:
    gr.Markdown("# 🌐 ZamAI Multilingual Embeddings")
    gr.Markdown("Generate embeddings and calculate similarity for multilingual text")
    
    with gr.Row():
        with gr.Column():
            text1 = gr.Textbox(
                label="First Text",
                placeholder="Enter text in any language..."
            )
            text2 = gr.Textbox(
                label="Second Text (optional, for similarity)",
                placeholder="Enter second text to compare..."
            )
            process_btn = gr.Button("Process", variant="primary")
        
        with gr.Column():
            embeddings_output = gr.Textbox(
                label="Embeddings",
                lines=5
            )
            similarity_output = gr.Number(
                label="Similarity Score",
                precision=4
            )
    
    process_btn.click(
        get_embeddings,
        inputs=[text1, text2],
        outputs=[embeddings_output, similarity_output]
    )

if __name__ == "__main__":
    demo.launch()
'''
    else:
        # Default fallback
        app_code = f'''import gradio as gr

def placeholder_function(input_text):
    return f"Model {model_id} is being set up. Please try again later."

demo = gr.Interface(
    fn=placeholder_function,
    inputs="text",
    outputs="text",
    title="ZamAI Model Interface"
)

if __name__ == "__main__":
    demo.launch()
'''
    
    return app_code

def create_requirements_txt():
    """Create requirements.txt for spaces"""
    return """gradio>=4.8.0
transformers>=4.30.0
torch>=2.0.0
sentence-transformers>=2.2.0
numpy>=1.21.0
"""

def create_spaces():
    """Create all Hugging Face Spaces for ZamAI models"""
    
    token = read_hf_token()
    api = HfApi(token=token)
    
    # Define spaces to create
    spaces_config = [
        {
            "space_name": "zamai-pashto-chat",
            "title": "ZamAI Pashto Chat",
            "emoji": "💬",
            "description": "Advanced conversational AI for Pashto language with cultural context understanding.",
            "model_id": "tasal9/ZamAI-LIama3-Pashto",
            "task_type": "text-generation"
        },
        {
            "space_name": "zamai-pashto-bloom",
            "title": "ZamAI Pashto Text Generator",
            "emoji": "📝",
            "description": "Generate high-quality Pashto text using BLOOM architecture fine-tuned for Afghan cultural context.",
            "model_id": "tasal9/pashto-base-bloom",
            "task_type": "text-generation"
        },
        {
            "space_name": "zamai-mistral-pashto",
            "title": "ZamAI Mistral Pashto",
            "emoji": "🚀",
            "description": "State-of-the-art Pashto text generation using Mistral-7B architecture.",
            "model_id": "tasal9/ZamAI-Mistral-7B-Pashto",
            "task_type": "text-generation"
        },
        {
            "space_name": "zamai-phi3-business",
            "title": "ZamAI Phi-3 Business Assistant",
            "emoji": "💼",
            "description": "Business automation and document processing for Pashto content.",
            "model_id": "tasal9/ZamAI-Phi-3-Mini-Pashto",
            "task_type": "text-generation"
        },
        {
            "space_name": "zamai-whisper-speech",
            "title": "ZamAI Pashto Speech Recognition",
            "emoji": "🎤",
            "description": "Convert Pashto speech to text with high accuracy using fine-tuned Whisper model.",
            "model_id": "tasal9/ZamAI-Whisper-v3-Pashto",
            "task_type": "automatic-speech-recognition"
        },
        {
            "space_name": "zamai-embeddings",
            "title": "ZamAI Multilingual Embeddings",
            "emoji": "🌐",
            "description": "Generate embeddings and calculate similarity for multilingual text including Pashto.",
            "model_id": "tasal9/Multilingual-ZamAI-Embeddings",
            "task_type": "feature-extraction"
        }
    ]
    
    created_spaces = []
    
    for space_config in spaces_config:
        space_name = space_config["space_name"]
        full_space_name = f"tasal9/{space_name}"
        
        print(f"\\n🚀 Creating space: {full_space_name}")
        
        try:
            # Create temporary directory for space files
            temp_dir = f"/tmp/{space_name}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create README.md
            readme_content = create_space_config(
                space_name,
                space_config["title"],
                space_config["emoji"],
                space_config["description"],
                space_config["model_id"]
            )
            
            with open(f"{temp_dir}/README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)
            
            # Create app.py
            app_content = create_basic_app(
                space_config["model_id"],
                space_config["task_type"]
            )
            
            with open(f"{temp_dir}/app.py", "w", encoding="utf-8") as f:
                f.write(app_content)
            
            # Create requirements.txt
            with open(f"{temp_dir}/requirements.txt", "w") as f:
                f.write(create_requirements_txt())
            
            # Create the space repository
            try:
                create_repo(
                    repo_id=full_space_name,
                    repo_type="space",
                    space_sdk="gradio",
                    token=token,
                    exist_ok=True
                )
                print(f"✅ Repository created: {full_space_name}")
            except Exception as e:
                print(f"⚠️  Repository might already exist: {e}")
            
            # Upload files to the space
            upload_folder(
                folder_path=temp_dir,
                repo_id=full_space_name,
                repo_type="space",
                token=token,
                commit_message=f"Initial setup for {space_config['title']}"
            )
            
            print(f"✅ Space created successfully: https://huggingface.co/spaces/{full_space_name}")
            created_spaces.append(full_space_name)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"❌ Error creating space {full_space_name}: {e}")
    
    return created_spaces

def create_training_space():
    """Create a special ZeroGPU training space"""
    token = read_hf_token()
    api = HfApi(token=token)
    
    space_name = "tasal9/zamai-training-hub"
    print(f"\\n🔥 Creating training space: {space_name}")
    
    try:
        # Copy the existing zerogpu files
        source_dir = "/workspaces/ZamAI-Pro-Models/zerogpu_files"
        temp_dir = "/tmp/zamai-training-hub"
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        shutil.copytree(source_dir, temp_dir)
        
        # Create the space repository
        try:
            create_repo(
                repo_id=space_name,
                repo_type="space",
                space_sdk="gradio",
                token=token,
                exist_ok=True
            )
            print(f"✅ Training space repository created: {space_name}")
        except Exception as e:
            print(f"⚠️  Repository might already exist: {e}")
        
        # Upload files to the space
        upload_folder(
            folder_path=temp_dir,
            repo_id=space_name,
            repo_type="space",
            token=token,
            commit_message="Setup ZamAI training hub with ZeroGPU"
        )
        
        print(f"✅ Training space created: https://huggingface.co/spaces/{space_name}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return space_name
        
    except Exception as e:
        print(f"❌ Error creating training space: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Setting up ZamAI Hugging Face Spaces...")
    print("=" * 50)
    
    # Create all model demo spaces
    created_spaces = create_spaces()
    
    # Create training space
    training_space = create_training_space()
    if training_space:
        created_spaces.append(training_space)
    
    print("\\n" + "=" * 50)
    print(f"✅ Setup Complete! Created {len(created_spaces)} spaces:")
    for space in created_spaces:
        print(f"   🌐 https://huggingface.co/spaces/{space}")
    
    print("\\n🎯 Next Steps:")
    print("1. Visit each space to test functionality")
    print("2. Customize the interfaces as needed")
    print("3. Use the training space to create new models")
    print("4. Share your spaces with the community!")
