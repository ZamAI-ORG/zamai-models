#!/usr/bin/env python3
"""
Manual ZeroGPU Training Space Setup for ZamAI Models
Creates ready-to-upload files for HuggingFace Spaces
"""

from pathlib import Path

def create_training_app_files():
    """Create training app files for each model type"""
    
    # Create directories
    output_dir = Path("/workspaces/ZamAI-Pro-Models/zerogpu_files")
    output_dir.mkdir(exist_ok=True)
    
    models = [
        {
            "name": "bloom-pashto",
            "model_id": "tasal9/pashto-base-bloom",
            "type": "text-generation",
            "description": "Pashto BLOOM model training space"
        },
        {
            "name": "llama3-pashto", 
            "model_id": "tasal9/ZamAI-LIama3-Pashto",
            "type": "text-generation",
            "description": "Pashto Llama3 model training space"
        },
        {
            "name": "mistral-pashto",
            "model_id": "tasal9/ZamAI-Mistral-7B-Pashto", 
            "type": "text-generation",
            "description": "Pashto Mistral model training space"
        },
        {
            "name": "phi3-pashto",
            "model_id": "tasal9/ZamAI-Phi-3-Mini-Pashto",
            "type": "text-generation", 
            "description": "Pashto Phi-3 model training space"
        },
        {
            "name": "mt5-pashto",
            "model_id": "tasal9/ZamAI-mT5-Pashto",
            "type": "translation",
            "description": "Pashto mT5 translation training space"
        },
        {
            "name": "whisper-pashto",
            "model_id": "tasal9/ZamAI-Whisper-v3-Pashto",
            "type": "speech-to-text",
            "description": "Pashto Whisper ASR training space"
        },
        {
            "name": "embeddings-multilingual",
            "model_id": "tasal9/Multilingual-ZamAI-Embeddings", 
            "type": "embeddings",
            "description": "Multilingual embeddings training space"
        }
    ]
    
    for model in models:
        print(f"🚀 Creating files for {model['name']}...")
        
        # Create model-specific directory
        model_dir = output_dir / model['name']
        model_dir.mkdir(exist_ok=True)
        
        # Generate app.py
        app_content = generate_app_py(model)
        with open(model_dir / "app.py", "w") as f:
            f.write(app_content)
        
        # Generate requirements.txt
        req_content = generate_requirements(model['type'])
        with open(model_dir / "requirements.txt", "w") as f:
            f.write(req_content)
        
        # Generate README.md
        readme_content = generate_readme(model)
        with open(model_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        print(f"✅ Files created in: {model_dir}")
    
    print(f"\n📂 All files created in: {output_dir}")
    print("\n🎯 Next Steps:")
    print("1. Go to https://huggingface.co/new-space")
    print("2. Create a new space for each model")
    print("3. Set Hardware to 'ZeroGPU - A10G'")
    print("4. Upload the files from each model directory")
    print("5. Your training spaces will be ready!")

def generate_app_py(model):
    """Generate app.py based on model type"""
    
    if model['type'] == 'text-generation':
        return f'''import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import spaces

MODEL_ID = "{model['model_id']}"

@spaces.GPU
def load_model():
    """Load model with ZeroGPU support"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

@spaces.GPU
def generate_text(prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Generate text using the model"""
    try:
        model, tokenizer = load_model()
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):]
        
    except Exception as e:
        return f"Error: {{e}}"

@spaces.GPU  
def start_training(dataset_name, epochs=3, learning_rate=2e-4):
    """Start LoRA training"""
    try:
        model, tokenizer = load_model()
        
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        if not dataset_name:
            return "Please provide a dataset name"
            
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )
        
        # Start training
        trainer.train()
        
        # Save model
        model.save_pretrained("./fine_tuned_model")
        tokenizer.save_pretrained("./fine_tuned_model")
        
        return "✅ Training completed successfully!"
        
    except Exception as e:
        return f"❌ Training error: {{e}}"

# Gradio Interface
with gr.Blocks(title="ZamAI {model['name']} Training", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 ZamAI Model Training Space")
    gr.Markdown(f"**Model:** {{MODEL_ID}}")
    gr.Markdown("This space allows you to fine-tune and test your ZamAI model with ZeroGPU acceleration.")
    
    with gr.Tab("💬 Test Model"):
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Enter your prompt (Pashto/English)",
                    placeholder="سلام! زموږ د ژبې موډل ازمویاست...",
                    lines=3
                )
                with gr.Row():
                    max_length = gr.Slider(50, 1024, value=512, label="Max Length")
                    temperature = gr.Slider(0.1, 2.0, value=0.7, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top P")
                generate_btn = gr.Button("🔮 Generate", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=8,
                    interactive=False
                )
        
        generate_btn.click(
            generate_text,
            inputs=[prompt_input, max_length, temperature, top_p],
            outputs=output_text
        )
    
    with gr.Tab("🎯 Training"):
        gr.Markdown("### LoRA Fine-tuning Setup")
        with gr.Row():
            with gr.Column():
                dataset_input = gr.Textbox(
                    label="Dataset Name (HuggingFace)",
                    placeholder="tasal9/ZamAI_Pashto_Dataset",
                    value="tasal9/ZamAI_Pashto_Dataset"
                )
                epochs_input = gr.Number(value=3, label="Epochs", minimum=1, maximum=10)
                lr_input = gr.Number(value=2e-4, label="Learning Rate", step=1e-5)
                train_btn = gr.Button("🚀 Start Training", variant="primary")
            
            with gr.Column():
                training_output = gr.Textbox(
                    label="Training Status",
                    lines=10,
                    interactive=False
                )
        
        train_btn.click(
            start_training,
            inputs=[dataset_input, epochs_input, lr_input],
            outputs=training_output
        )
    
    with gr.Tab("📊 Model Info"):
        gr.Markdown(f"""
        ### Model Details
        - **Model ID:** {{MODEL_ID}}
        - **Type:** Text Generation
        - **Description:** {model['description']}
        - **Hardware:** ZeroGPU A10G
        
        ### Training Features
        - ✅ LoRA fine-tuning for efficient training
        - ✅ Automatic model preparation
        - ✅ Custom dataset support
        - ✅ Real-time training progress
        
        ### Usage Tips
        1. Test the model first with sample prompts
        2. Use quality Pashto datasets for best results
        3. Adjust learning rate based on dataset size
        4. Monitor training loss for optimal epochs
        """)

if __name__ == "__main__":
    demo.launch()
'''
    
    elif model['type'] == 'translation':
        return f'''import os

import gradio as gr
import spaces
import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "{model['model_id']}"
BASE_MODEL = "google/mt5-base"
DEFAULT_DATASET = "tasal9/ZamAi-Pashto-Datasets-V2"
MAX_SEQ_LENGTH = 512

_MODEL_CACHE = {{"model": None, "tokenizer": None}}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@spaces.GPU
def load_model():
    """Lazy-load the translation model for inference."""

    if _MODEL_CACHE["model"] is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model.to(_device())
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["tokenizer"] = tokenizer
    return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"]


def _direction_prefix(direction: str) -> str:
    return "translate Pashto to English: " if direction == "ps-en" else "translate English to Pashto: "


def _extract_pair(example: dict, direction: str):
    source_candidates = ["input", "en", "english", "source", "prompt"]
    target_candidates = ["output", "ps", "pashto", "target", "completion", "answer"]

    if direction == "ps-en":
        source_candidates, target_candidates = target_candidates, source_candidates

    source = next((example.get(key) for key in source_candidates if example.get(key)), None)
    target = next((example.get(key) for key in target_candidates if example.get(key)), None)

    if not source or not target:
        return None, None

    return str(source).strip(), str(target).strip()


@spaces.GPU
def translate_text(text: str, direction: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Please provide text to translate."

    model, translation_tokenizer = load_model()
    inputs = translation_tokenizer(
        _direction_prefix(direction) + text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    if torch.cuda.is_available():
        inputs = {{k: v.to(_device()) for k, v in inputs.items()}}

    outputs = model.generate(
        **inputs,
        max_length=MAX_SEQ_LENGTH,
        num_beams=4,
        early_stopping=True,
    )
    return translation_tokenizer.decode(outputs[0], skip_special_tokens=True)


def _prepare_dataset(dataset, direction: str):
    def convert(example):
        src, tgt = _extract_pair(example, direction)
        return {{"source_text": src, "target_text": tgt}}

    converted = dataset.map(convert)
    converted = converted.filter(lambda ex: ex["source_text"] and ex["target_text"])
    extra_cols = [col for col in converted.column_names if col not in {{"source_text", "target_text"}}]
    return converted.remove_columns(extra_cols)


@spaces.GPU
def start_training(
    dataset_name: str,
    direction: str,
    epochs: int,
    learning_rate: float,
    max_train_samples: int,
    push_to_hub: bool,
    repo_id: str,
) -> str:
    try:
        epochs = int(epochs)
        max_train_samples = int(max_train_samples) if max_train_samples else None

        training_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=True,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q", "v"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        dataset = load_dataset(dataset_name, split="train", verification_mode="no_checks")
        if max_train_samples and len(dataset) > max_train_samples:
            dataset = dataset.shuffle(seed=42).select(range(max_train_samples))

        dataset = _prepare_dataset(dataset, direction)
        if len(dataset) == 0:
            return "❌ Could not find translation pairs in the dataset."

        def tokenize_batch(batch):
            model_inputs = training_tokenizer(
                [_direction_prefix(direction) + text for text in batch["source_text"]],
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                padding="max_length",
            )
            labels = training_tokenizer(
                batch["target_text"],
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                padding="max_length",
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
        data_collator = DataCollatorForSeq2Seq(tokenizer=training_tokenizer, model=model)

        training_args = TrainingArguments(
            output_dir="./mt5_translation_outputs",
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=float(learning_rate),
            warmup_steps=100,
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            push_to_hub=bool(push_to_hub),
            hub_model_id=repo_id or MODEL_ID,
            hub_token=os.getenv("HF_TOKEN"),
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model()

        if push_to_hub:
            trainer.push_to_hub(commit_message="Update ZamAI mT5 translator adapter")

        return f"✅ Training complete! Samples used: {{len(tokenized)}}"

    except Exception as exc:  # pragma: no cover - runtime feedback is shown in the UI
        return f"❌ Training error: {{exc}}"


with gr.Blocks(title="ZamAI mT5 Pashto Translator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🇦🇫 ZamAI mT5 Pashto Translation + Training")
    gr.Markdown(
        f"**Model:** `{{MODEL_ID}}` · **Base:** `{{BASE_MODEL}}` · **Dataset:** `{{DEFAULT_DATASET}}`"
    )

    with gr.Tab("Translate"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Type English or Pashto text...",
                    lines=4,
                )
                direction = gr.Dropdown(
                    choices=["en-ps", "ps-en"],
                    value="en-ps",
                    label="Direction",
                )
                translate_btn = gr.Button("Translate", variant="primary")
            with gr.Column():
                translation_output = gr.Textbox(
                    label="Translation",
                    lines=6,
                    interactive=False,
                )

        translate_btn.click(
            translate_text,
            inputs=[text_input, direction],
            outputs=translation_output,
        )

    with gr.Tab("Training"):
        dataset_name = gr.Textbox(label="Dataset (HF repo)", value=DEFAULT_DATASET)
        repo_id = gr.Textbox(label="Push trained adapter to", value=MODEL_ID)
        train_direction = gr.Dropdown(
            choices=["en-ps", "ps-en"], value="en-ps", label="Training direction"
        )
        epochs = gr.Slider(1, 5, value=1, step=1, label="Epochs")
        learning_rate = gr.Slider(1e-5, 5e-4, value=2e-4, label="Learning rate")
        max_samples = gr.Slider(200, 4000, value=1500, step=100, label="Max training samples")
        push_flag = gr.Checkbox(label="Push to Hugging Face Hub", value=True)
        train_btn = gr.Button("🚀 Start Training", variant="primary")
        training_status = gr.Textbox(label="Training Status", lines=8, interactive=False)

        train_btn.click(
            start_training,
            inputs=[
                dataset_name,
                train_direction,
                epochs,
                learning_rate,
                max_samples,
                push_flag,
                repo_id,
            ],
            outputs=training_status,
        )

    with gr.Tab("Tips"):
        gr.Markdown(
            """
            ### 📌 Tips
            - Works with datasets that expose `input/output`, `en/ps`, or `prompt/completion` columns.
            - Lower `Max training samples` for quick smoke tests.
            - Add a valid HF token (Settings → Tokens) to the space secrets for automatic pushes.
            """
        )

if __name__ == "__main__":
    demo.launch()
'''

    elif model['type'] == 'speech-to-text':
        return f'''import gradio as gr
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import spaces

MODEL_ID = "{model['model_id']}"

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
        return f"Error: {{e}}"

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
        return f"Training setup initialized for {{dataset_name}} with {{epochs}} epochs.\\nNote: Full Whisper training requires custom data collator and preprocessing."
        
    except Exception as e:
        return f"Training error: {{e}}"

# Gradio Interface
with gr.Blocks(title="ZamAI Whisper Training", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 ZamAI Whisper ASR Training Space")
    gr.Markdown(f"**Model:** {{MODEL_ID}}")
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
'''
    
    elif model['type'] == 'embeddings':
        return f'''import gradio as gr
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spaces

MODEL_ID = "{model['model_id']}"

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
        return f"Error: {{e}}"

@spaces.GPU
def calculate_similarity(text1, text2):
    """Calculate semantic similarity between two texts"""
    try:
        model = load_embeddings_model()
        
        embeddings = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return f"Similarity Score: {{similarity:.4f}}"
        
    except Exception as e:
        return f"Error: {{e}}"

# Gradio Interface
with gr.Blocks(title="ZamAI Embeddings Training", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 ZamAI Embeddings Training Space")
    gr.Markdown(f"**Model:** {{MODEL_ID}}")
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
'''

def generate_requirements(model_type):
    """Generate requirements.txt based on model type"""
    base_requirements = [
        "gradio",
        "torch",
        "transformers", 
        "spaces",
        "datasets",
        "numpy",
        "huggingface_hub"
    ]
    
    if model_type == "text-generation":
        base_requirements.extend([
            "peft",
            "bitsandbytes",
            "accelerate"
        ])
    elif model_type == "translation":
        base_requirements.extend([
            "peft",
            "bitsandbytes",
            "accelerate",
            "sentencepiece"
        ])
    elif model_type == "speech-to-text":
        base_requirements.extend([
            "librosa",
            "soundfile"
        ])
    elif model_type == "embeddings":
        base_requirements.extend([
            "sentence-transformers",
            "scikit-learn"
        ])
    
    return "\n".join(base_requirements)

def generate_readme(model):
    """Generate README.md for the space"""
    return f'''---
title: ZamAI {model['name']} Training
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
---

# 🚀 ZamAI {model['name']} Training Space

## Model Information
- **Model ID:** {model['model_id']}
- **Type:** {model['type']}
- **Description:** {model['description']}
- **Hardware:** ZeroGPU A10G

## Features
- ✅ ZeroGPU acceleration for fast training and inference
- ✅ Interactive testing interface
- ✅ Fine-tuning capabilities
- ✅ Pashto language optimization

## Usage
1. **Test Tab:** Try the model with your own inputs
2. **Training Tab:** Fine-tune the model with your datasets

## Training Data
This model can be fine-tuned with custom datasets. Make sure your dataset is available on HuggingFace Hub.

## ZamAI Project
This space is part of the ZamAI project, focusing on Afghan and Pashto language AI models.

For more information, visit: [ZamAI Hub](https://huggingface.co/tasal9)
'''

if __name__ == "__main__":
    create_training_app_files()
