#!/usr/bin/env python3
"""
Test the fixed Whisper model and create a working Whisper space
"""

import os
from huggingface_hub import HfApi, create_repo, upload_file
import tempfile

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def test_whisper_model():
    """Test if the Whisper model works now"""
    
    try:
        from transformers import pipeline
        
        print("🧪 Testing Whisper model loading...")
        
        # Test primary model
        try:
            transcriber = pipeline("automatic-speech-recognition", model="tasal9/ZamAI-Whisper-v3-Pashto")
            print("✅ Primary Whisper model loads successfully!")
            return True
        except Exception as e:
            print(f"❌ Primary model failed: {str(e)[:100]}")
            
            # Test fallback
            try:
                transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
                print("✅ Fallback Whisper model works!")
                return False  # Primary failed but fallback works
            except Exception as e2:
                print(f"❌ Both models failed: {str(e2)[:100]}")
                return False
                
    except ImportError:
        print("⚠️  Could not import transformers for testing")
        return False

def create_working_whisper_space():
    """Create a robust Whisper space that handles model issues"""
    
    token = read_hf_token()
    space_id = "tasal9/zamai-whisper-speech-robust"
    
    print(f"🚀 Creating robust Whisper space: {space_id}")
    
    # README.md
    readme = '''---
title: ZamAI Whisper Speech (Robust)
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
models:
- tasal9/ZamAI-Whisper-v3-Pashto
- openai/whisper-base
---

# 🎤 ZamAI Whisper Speech Recognition (Robust)

Convert Pashto speech to text with high accuracy using fine-tuned Whisper model.
Includes automatic fallback to ensure reliability.

## Features
- ✅ Primary: Fine-tuned Pashto Whisper model
- ✅ Fallback: Official Whisper base model  
- ✅ ZeroGPU acceleration
- ✅ Robust error handling

## Usage
1. Click "Load Model" to initialize
2. Record audio or upload file
3. Click "Transcribe" to get text output
'''

    # app.py
    app = '''import gradio as gr
from transformers import pipeline
import torch
import spaces

# Model configuration
PRIMARY_MODEL = "tasal9/ZamAI-Whisper-v3-Pashto"
FALLBACK_MODEL = "openai/whisper-base"
transcriber = None
current_model = None

@spaces.GPU
def load_model():
    global transcriber, current_model
    
    # Try primary model first
    try:
        transcriber = pipeline(
            "automatic-speech-recognition", 
            model=PRIMARY_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        current_model = PRIMARY_MODEL
        return f"✅ Successfully loaded: {PRIMARY_MODEL}"
    except Exception as e:
        print(f"Primary model failed: {e}")
        
        # Fallback to official Whisper
        try:
            transcriber = pipeline(
                "automatic-speech-recognition",
                model=FALLBACK_MODEL,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            current_model = FALLBACK_MODEL
            return f"⚠️ Using fallback model: {FALLBACK_MODEL}\\n(Primary model issue: {str(e)[:100]}...)"
        except Exception as e2:
            transcriber = None
            current_model = None
            return f"❌ All models failed to load:\\nPrimary: {str(e)[:50]}\\nFallback: {str(e2)[:50]}"

@spaces.GPU
def transcribe_audio(audio):
    if transcriber is None:
        return "❌ Please load the model first using the 'Load Model' button."
    
    if audio is None:
        return "❌ Please provide audio input (record or upload a file)."
    
    try:
        result = transcriber(audio)
        text = result["text"]
        
        # Add model info to result
        model_info = f"\\n\\n📄 Transcribed using: {current_model}"
        return text + model_info
        
    except Exception as e:
        return f"❌ Transcription error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="ZamAI Whisper Speech Recognition") as demo:
    gr.Markdown("# 🎤 ZamAI Whisper Speech Recognition")
    gr.Markdown("Convert Pashto speech to text with automatic fallback for reliability")
    
    with gr.Row():
        load_btn = gr.Button("🔄 Load Model", variant="secondary", size="lg")
        status = gr.Textbox(
            label="Model Status", 
            value="Click 'Load Model' to initialize speech recognition",
            lines=3
        )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎙️ Audio Input")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or Upload Audio",
                format="wav"
            )
            
            transcribe_btn = gr.Button("📝 Transcribe Speech", variant="primary", size="lg")
            
            gr.Markdown("### 💡 Tips")
            gr.Markdown("""
            - For best results, use clear audio with minimal background noise
            - Pashto speech works best with the primary model
            - Other languages will use the fallback Whisper model
            - Supported formats: WAV, MP3, M4A, FLAC
            """)
        
        with gr.Column():
            gr.Markdown("### 📄 Transcription Results")
            output = gr.Textbox(
                label="Transcribed Text",
                lines=10,
                placeholder="Transcription will appear here..."
            )
    
    # Event handlers
    load_btn.click(load_model, outputs=status)
    transcribe_btn.click(
        transcribe_audio, 
        inputs=audio_input, 
        outputs=output
    )
    
    # Auto-transcribe on audio upload
    audio_input.change(
        transcribe_audio,
        inputs=audio_input,
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
'''

    # requirements.txt
    requirements = '''gradio>=4.8.0
transformers>=4.30.0
torch>=2.0.0
accelerate>=0.20.0
spaces
librosa
soundfile
'''

    try:
        # Create repository
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            token=token,
            exist_ok=True
        )
        print("✅ Repository created")
        
        # Upload files
        files_to_upload = [
            ("README.md", readme),
            ("app.py", app),
            ("requirements.txt", requirements)
        ]
        
        for filename, content in files_to_upload:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(content)
                temp_path = f.name
            
            upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=filename,
                repo_id=space_id,
                repo_type="space",
                token=token,
                commit_message=f"Add {filename} for robust Whisper space"
            )
            
            os.unlink(temp_path)
            print(f"✅ {filename} uploaded")
        
        print(f"🌐 Robust Whisper space created: https://huggingface.co/spaces/{space_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating space: {e}")
        return False

def main():
    print("🧪 TESTING WHISPER MODEL & CREATING ROBUST SPACE")
    print("=" * 60)
    
    # Test the models
    model_works = test_whisper_model()
    
    # Create a robust space regardless
    print("\\n🚀 Creating robust Whisper space...")
    space_created = create_working_whisper_space()
    
    print("\\n" + "=" * 60)
    print("📋 SUMMARY:")
    
    if model_works:
        print("✅ Primary Whisper model is working")
    else:
        print("⚠️  Primary Whisper model has issues - using fallback")
    
    if space_created:
        print("✅ Robust Whisper space created successfully")
        print("🎯 This space will work regardless of model issues")
    
    print("\\n💡 RECOMMENDATIONS:")
    print("1. Use the robust space for reliable speech recognition")
    print("2. Consider re-training/uploading the Whisper model properly")
    print("3. The space will automatically use fallback if primary model fails")

if __name__ == "__main__":
    main()
