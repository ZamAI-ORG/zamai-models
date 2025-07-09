import gradio as gr
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import spaces

MODEL_ID = "tasal9/ZamAI-Whisper-v3-Pashto"

@spaces.GPU
def load_whisper_model():
    """Load Whisper model with ZeroGPU support"""
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

@spaces.GPU
def transcribe_audio(audio_file, language="ps"):
    """Transcribe audio using Whisper model"""
    try:
        model, processor = load_whisper_model()
        
        # Load audio
        if audio_file is None:
            return "Please upload an audio file"
            
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Process audio
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"])
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
        
    except Exception as e:
        return f"Error: {e}"

@spaces.GPU
def fine_tune_whisper(dataset_name, epochs=3):
    """Fine-tune Whisper model"""
    try:
        model, processor = load_whisper_model()
        
        if not dataset_name:
            return "Please provide an audio dataset name"
        
        # Load audio dataset
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        # This is a simplified setup - full Whisper training requires more complex preprocessing
        return f"Training setup initialized for {dataset_name} with {epochs} epochs.\nNote: Full Whisper training requires custom data collator and preprocessing."
        
    except Exception as e:
        return f"Training error: {e}"

# Gradio Interface
with gr.Blocks(title="ZamAI Whisper Training", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 ZamAI Whisper ASR Training Space")
    gr.Markdown(f"**Model:** {MODEL_ID}")
    gr.Markdown("This space allows you to test and fine-tune your Whisper model for Pashto speech recognition.")
    
    with gr.Tab("🎵 Test ASR"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath"
                )
                language_select = gr.Dropdown(
                    choices=["ps", "en", "fa", "ar", "ur"],
                    value="ps",
                    label="Language"
                )
                transcribe_btn = gr.Button("🎯 Transcribe", variant="primary")
            
            with gr.Column():
                transcription_output = gr.Textbox(
                    label="Transcription",
                    lines=8,
                    interactive=False
                )
        
        transcribe_btn.click(
            transcribe_audio,
            inputs=[audio_input, language_select],
            outputs=transcription_output
        )
    
    with gr.Tab("🎯 Training"):
        gr.Markdown("### Whisper Fine-tuning Setup")
        with gr.Row():
            with gr.Column():
                dataset_input = gr.Textbox(
                    label="Audio Dataset Name",
                    placeholder="tasal9/ZamAI_Pashto_Speech_Dataset",
                    value="tasal9/ZamAI_Pashto_Speech_Dataset"
                )
                epochs_input = gr.Number(value=3, label="Epochs")
                train_btn = gr.Button("🚀 Start Training", variant="primary")
            
            with gr.Column():
                training_output = gr.Textbox(
                    label="Training Status", 
                    lines=10,
                    interactive=False
                )
        
        train_btn.click(
            fine_tune_whisper,
            inputs=[dataset_input, epochs_input],
            outputs=training_output
        )

if __name__ == "__main__":
    demo.launch()
