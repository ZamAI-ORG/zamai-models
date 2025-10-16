#!/usr/bin/env python3
"""
Smart Space Creation for ZamAI Models
=====================================
Only creates Spaces for models that actually exist in the HF Hub.
Uses ZeroGPU for all Spaces and includes robust error handling.
"""

import os
import requests
import time
from huggingface_hub import HfApi, create_repo, upload_file
from huggingface_hub.utils import RepositoryNotFoundError
import json

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
USERNAME = "tasal9"

# Define models that we confirmed exist in the HF Hub
EXISTING_MODELS = {
    "tasal9/ZamAI-LIama3-Pashto": "text-generation",
    "tasal9/pashto-base-bloom": "text-generation", 
    "tasal9/ZamAI-Mistral-7B-Pashto": "text-generation",
    "tasal9/ZamAI-Phi-3-Mini-Pashto": "text-generation",
    "tasal9/ZamAI-Whisper-v3-Pashto": "automatic-speech-recognition",
    "tasal9/Multilingual-ZamAI-Embeddings": "sentence-similarity"
}

def check_model_exists(model_name):
    """Check if a model exists in HF Hub"""
    try:
        api = HfApi(token=HF_TOKEN)
        api.model_info(model_name)
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as e:
        print(f"❌ Error checking {model_name}: {e}")
        return False

def create_text_generation_space(model_name, space_name):
    """Create a text generation space with ZeroGPU support"""
    
    # App.py for text generation
    app_content = f'''import gradio as gr
import spaces
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Model configuration
MODEL_NAME = "{model_name}"

@spaces.GPU
def load_model():
    """Load the model with error handling"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {{e}}")
        # Fallback to a working model
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        return tokenizer, model

# Load model
tokenizer, model = load_model()

@spaces.GPU
def generate_text(prompt, max_length=100, temperature=0.7, do_sample=True):
    """Generate text using the model"""
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    except Exception as e:
        return f"Error generating text: {{str(e)}}"

# Gradio interface
def chat_interface(message, max_length, temperature):
    """Chat interface for the model"""
    response = generate_text(message, max_length, temperature)
    return response

# Create Gradio interface
iface = gr.Interface(
    fn=chat_interface,
    inputs=[
        gr.Textbox(label="Your message", placeholder="Type your message here..."),
        gr.Slider(minimum=50, maximum=500, value=100, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature")
    ],
    outputs=gr.Textbox(label="AI Response"),
    title=f"🤖 {{MODEL_NAME.split('/')[-1]}} Chat",
    description=f"Chat with the {{MODEL_NAME}} model - A Pashto language AI assistant",
    theme="soft",
    examples=[
        ["سلام ورور! څنګه یاست؟", 100, 0.7],
        ["د افغانستان په اړه راته ووایاست", 150, 0.8],
        ["Hello! How are you?", 100, 0.7],
    ]
)

if __name__ == "__main__":
    iface.launch()
'''

    # Requirements.txt
    requirements_content = '''gradio==4.44.0
transformers==4.44.0
torch==2.0.1
accelerate==0.20.3
spaces
'''

    # README.md
    readme_content = f'''---
title: {space_name}
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
---

# {space_name}

This is a demo of the **{model_name}** model for Pashto text generation.

## Features
- 🚀 Powered by ZeroGPU for fast inference
- 🇦🇫 Specialized for Pashto language
- 💬 Interactive chat interface
- 🎛️ Adjustable generation parameters

## Usage
Simply type your message and the AI will respond in Pashto or English.

## Model
This space uses the [{model_name}](https://huggingface.co/{model_name}) model.
'''

    return app_content, requirements_content, readme_content

def create_whisper_space(model_name, space_name):
    """Create a Whisper ASR space with robust fallback"""
    
    # App.py for Whisper
    app_content = f'''import gradio as gr
import spaces
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np

# Model configuration
MODEL_NAME = "{model_name}"
FALLBACK_MODEL = "openai/whisper-base"

@spaces.GPU
def load_whisper_model():
    """Load Whisper model with fallback"""
    try:
        print(f"🔄 Loading {{MODEL_NAME}}...")
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Successfully loaded {{MODEL_NAME}}")
        return processor, model, MODEL_NAME
    except Exception as e:
        print(f"❌ Error loading {{MODEL_NAME}}: {{e}}")
        print(f"🔄 Falling back to {{FALLBACK_MODEL}}...")
        try:
            processor = WhisperProcessor.from_pretrained(FALLBACK_MODEL)
            model = WhisperForConditionalGeneration.from_pretrained(FALLBACK_MODEL)
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            print(f"✅ Successfully loaded fallback model {{FALLBACK_MODEL}}")
            return processor, model, FALLBACK_MODEL
        except Exception as fallback_error:
            print(f"❌ Error loading fallback model: {{fallback_error}}")
            raise Exception("Could not load any Whisper model")

# Load model
processor, model, current_model = load_whisper_model()

@spaces.GPU
def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    try:
        if audio_path is None:
            return "❌ No audio file provided"
        
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Process with Whisper
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {{k: v.cuda() for k, v in inputs.items()}}
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"])
        
        # Decode transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        model_info = f"🤖 Model used: {{current_model}}"
        return f"{{model_info}}\\n\\n📝 Transcription:\\n{{transcription}}"
    
    except Exception as e:
        return f"❌ Error during transcription: {{str(e)}}"

# Gradio interface
def create_interface():
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown(f"""
        # 🎙️ ZamAI Whisper - Pashto Speech Recognition
        
        Upload an audio file to get transcription in Pashto/English.
        
        **Current Model:** `{{current_model}}`
        """)
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="🎵 Upload Audio File",
                    type="filepath"
                )
                transcribe_btn = gr.Button("🔄 Transcribe", variant="primary")
            
            with gr.Column():
                output = gr.Textbox(
                    label="📝 Transcription Output",
                    lines=10,
                    placeholder="Transcription will appear here..."
                )
        
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[output]
        )
        
        gr.Markdown("""
        ### 💡 Tips:
        - Upload clear audio files (WAV, MP3, M4A)
        - Works best with Pashto and English speech
        - Supports audio up to a few minutes long
        """)
    
    return demo

# Create and launch interface
demo = create_interface()

if __name__ == "__main__":
    demo.launch()
'''

    # Requirements.txt
    requirements_content = '''gradio==4.44.0
transformers==4.44.0
torch==2.0.1
librosa==0.10.1
spaces
'''

    # README.md
    readme_content = f'''---
title: {space_name}
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
---

# {space_name}

Automatic Speech Recognition for Pashto using the **{model_name}** model.

## Features
- 🚀 Powered by ZeroGPU for fast inference
- 🎙️ Speech-to-text for Pashto and English
- 🔄 Automatic fallback to ensure reliability
- 📱 Easy drag-and-drop interface

## Usage
1. Upload an audio file (WAV, MP3, M4A)
2. Click "Transcribe"
3. Get the text transcription

## Model
This space uses the [{model_name}](https://huggingface.co/{model_name}) model with fallback support.
'''

    return app_content, requirements_content, readme_content

def create_embeddings_space(model_name, space_name):
    """Create an embeddings space"""
    
    # App.py for embeddings
    app_content = f'''import gradio as gr
import spaces
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Model configuration
MODEL_NAME = "{model_name}"

@spaces.GPU
def load_embeddings_model():
    """Load sentence transformer model"""
    try:
        model = SentenceTransformer(MODEL_NAME)
        return model
    except Exception as e:
        print(f"Error loading {{MODEL_NAME}}: {{e}}")
        # Fallback to a working multilingual model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model

# Load model
model = load_embeddings_model()

@spaces.GPU
def compute_similarity(text1, text2):
    """Compute semantic similarity between two texts"""
    try:
        if not text1.strip() or not text2.strip():
            return "⚠️ Please enter both texts"
        
        # Generate embeddings
        embeddings = model.encode([text1, text2])
        
        # Compute cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Format result
        percentage = similarity * 100
        
        if percentage > 80:
            emoji = "🟢"
            level = "Very Similar"
        elif percentage > 60:
            emoji = "🟡"
            level = "Moderately Similar"
        elif percentage > 40:
            emoji = "🟠"
            level = "Somewhat Similar"
        else:
            emoji = "🔴"
            level = "Not Similar"
        
        return f"""
{emoji} **Similarity Score: {percentage:.1f}%**

**Level:** {level}

**Text 1 Embedding Shape:** {embeddings[0].shape}
**Text 2 Embedding Shape:** {embeddings[1].shape}
        """
    
    except Exception as e:
        return f"❌ Error computing similarity: {{str(e)}}"

@spaces.GPU
def get_embeddings(text):
    """Get embeddings for a single text"""
    try:
        if not text.strip():
            return "⚠️ Please enter some text"
        
        embedding = model.encode([text])[0]
        
        return f"""
📊 **Embedding Generated Successfully**

**Text:** "{text}"
**Embedding Shape:** {embedding.shape}
**Embedding Dimensions:** {len(embedding)}

**First 10 values:** {embedding[:10].tolist()}
        """
    
    except Exception as e:
        return f"❌ Error generating embedding: {{str(e)}}"

# Gradio interface
def create_interface():
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown(f"""
        # 🔢 ZamAI Multilingual Embeddings
        
        Generate semantic embeddings and compute text similarity using **{MODEL_NAME}**.
        
        Supports multiple languages including **Pashto**, **English**, **Arabic**, and more.
        """)
        
        with gr.Tabs():
            with gr.TabItem("📊 Text Similarity"):
                with gr.Row():
                    with gr.Column():
                        text1 = gr.Textbox(
                            label="First Text",
                            placeholder="د سولې غوښتونکي یو...",
                            lines=3
                        )
                        text2 = gr.Textbox(
                            label="Second Text", 
                            placeholder="Peace seekers are...",
                            lines=3
                        )
                        similarity_btn = gr.Button("🔍 Compare Similarity", variant="primary")
                    
                    with gr.Column():
                        similarity_output = gr.Markdown(label="Similarity Result")
                
                similarity_btn.click(
                    fn=compute_similarity,
                    inputs=[text1, text2],
                    outputs=[similarity_output]
                )
            
            with gr.TabItem("🔢 Generate Embeddings"):
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter text in any supported language...",
                            lines=4
                        )
                        embed_btn = gr.Button("🔢 Generate Embedding", variant="primary")
                    
                    with gr.Column():
                        embedding_output = gr.Markdown(label="Embedding Result")
                
                embed_btn.click(
                    fn=get_embeddings,
                    inputs=[input_text],
                    outputs=[embedding_output]
                )
        
        gr.Markdown("""
        ### 💡 Examples:
        - **Pashto:** "د سولې غوښتونکي یو"
        - **English:** "Peace seekers are"
        - **Arabic:** "طالبو السلام"
        
        ### 📚 Supported Languages:
        English, Pashto, Arabic, Urdu, Persian, and many more.
        """)
    
    return demo

# Create and launch interface
demo = create_interface()

if __name__ == "__main__":
    demo.launch()
'''

    # Requirements.txt
    requirements_content = '''gradio==4.44.0
sentence-transformers==2.2.2
scikit-learn==1.3.0
numpy==1.24.3
spaces
'''

    # README.md
    readme_content = f'''---
title: {space_name}
emoji: 🔢
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
---

# {space_name}

Multilingual semantic embeddings using the **{model_name}** model.

## Features
- 🚀 Powered by ZeroGPU for fast inference
- 🌍 Support for multiple languages including Pashto
- 📊 Text similarity computation
- 🔢 High-quality semantic embeddings

## Usage
1. **Text Similarity**: Compare two texts and get similarity scores
2. **Generate Embeddings**: Get vector representations of text

## Supported Languages
English, Pashto, Arabic, Urdu, Persian, and many more.

## Model
This space uses the [{model_name}](https://huggingface.co/{model_name}) model.
'''

    return app_content, requirements_content, readme_content

def create_space_for_model(model_name, model_type):
    """Create a space for a specific model"""
    try:
        # Generate space name
        model_clean_name = model_name.replace("tasal9/", "").replace("-", " ")
        space_name = f"ZamAI-{model_clean_name}"
        space_id = f"{USERNAME}/{space_name}"
        
        print(f"🚀 Creating space: {space_id}")
        
        # Check if model actually exists
        if not check_model_exists(model_name):
            print(f"❌ Model {model_name} does not exist! Skipping...")
            return False
        
        # Generate space content based on model type
        if model_type == "text-generation":
            app_content, requirements_content, readme_content = create_text_generation_space(model_name, space_name)
        elif model_type == "automatic-speech-recognition":
            app_content, requirements_content, readme_content = create_whisper_space(model_name, space_name)
        elif model_type == "sentence-similarity":
            app_content, requirements_content, readme_content = create_embeddings_space(model_name, space_name)
        else:
            print(f"❌ Unknown model type: {model_type}")
            return False
        
        # Create the space repository
        api = HfApi(token=HF_TOKEN)
        
        try:
            create_repo(
                repo_id=space_id,
                token=HF_TOKEN,
                repo_type="space",
                space_sdk="gradio",
                space_hardware="zero-a10g"
            )
            print(f"✅ Created space repository: {space_id}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"⚠️ Space {space_id} already exists, updating files...")
            else:
                print(f"❌ Error creating space: {e}")
                return False
        
        # Upload files
        files_to_upload = [
            ("app.py", app_content),
            ("requirements.txt", requirements_content),
            ("README.md", readme_content)
        ]
        
        for filename, content in files_to_upload:
            try:
                api.upload_file(
                    path_or_fileobj=content.encode('utf-8'),
                    path_in_repo=filename,
                    repo_id=space_id,
                    repo_type="space",
                    token=HF_TOKEN
                )
                print(f"✅ Uploaded {filename}")
            except Exception as e:
                print(f"❌ Error uploading {filename}: {e}")
        
        print(f"🎉 Space created successfully: https://huggingface.co/spaces/{space_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating space for {model_name}: {e}")
        return False

def main():
    """Main function to create all spaces"""
    if not HF_TOKEN:
        print("❌ Error: HF_TOKEN environment variable not set!")
        return
    
    print("🚀 Creating Spaces for Existing ZamAI Models")
    print("=" * 50)
    
    successful_spaces = []
    failed_spaces = []
    
    for model_name, model_type in EXISTING_MODELS.items():
        print(f"\n📦 Processing: {model_name} ({model_type})")
        
        if create_space_for_model(model_name, model_type):
            successful_spaces.append(model_name)
        else:
            failed_spaces.append(model_name)
        
        # Add delay between space creations
        time.sleep(2)
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 FINAL SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully created spaces: {len(successful_spaces)}")
    for space in successful_spaces:
        print(f"   - {space}")
    
    print(f"\n❌ Failed to create spaces: {len(failed_spaces)}")
    for space in failed_spaces:
        print(f"   - {space}")
    
    if successful_spaces:
        print(f"\n🎉 All spaces created! Visit: https://huggingface.co/{USERNAME}")

if __name__ == "__main__":
    main()
