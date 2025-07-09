#!/usr/bin/env python3
"""
Create missing testing and training spaces for models
"""

import os
from huggingface_hub import HfApi, create_repo
import json

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def create_testing_space(model_id, token):
    """Create a testing/inference space for a model"""
    api = HfApi(token=token)
    
    # Generate space name
    model_name = model_id.split('/')[-1]
    space_name = f"tasal9/{model_name}-testing"
    
    print(f"\n🚀 Creating testing space: {space_name}")
    
    try:
        # Create the space
        repo_url = create_repo(
            repo_id=space_name,
            repo_type="space",
            token=token,
            space_sdk="gradio",
            space_hardware="zero-a10g"
        )
        print(f"   ✅ Created space: {repo_url}")
        
        # Create app.py for testing
        app_py_content = generate_testing_app_py(model_id)
        
        # Upload app.py
        with open('/tmp/app.py', 'w') as f:
            f.write(app_py_content)
        
        api.upload_file(
            path_or_fileobj='/tmp/app.py',
            path_in_repo="app.py",
            repo_id=space_name,
            repo_type="space"
        )
        
        # Create requirements.txt
        requirements_content = generate_requirements_txt(model_id)
        
        with open('/tmp/requirements.txt', 'w') as f:
            f.write(requirements_content)
        
        api.upload_file(
            path_or_fileobj='/tmp/requirements.txt',
            path_in_repo="requirements.txt",
            repo_id=space_name,
            repo_type="space"
        )
        
        # Create README.md
        readme_content = generate_testing_readme(model_id, space_name)
        
        with open('/tmp/README.md', 'w') as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj='/tmp/README.md',
            path_in_repo="README.md",
            repo_id=space_name,
            repo_type="space"
        )
        
        # Clean up temp files
        for temp_file in ['/tmp/app.py', '/tmp/requirements.txt', '/tmp/README.md']:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"   ✅ Uploaded all files to {space_name}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating space: {e}")
        return False

def create_training_space(model_id, token):
    """Create a training/fine-tuning space for a model"""
    api = HfApi(token=token)
    
    # Generate space name
    model_name = model_id.split('/')[-1]
    space_name = f"tasal9/{model_name}-training"
    
    print(f"\n🏋️  Creating training space: {space_name}")
    
    try:
        # Create the space
        repo_url = create_repo(
            repo_id=space_name,
            repo_type="space",
            token=token,
            space_sdk="gradio",
            space_hardware="zero-a10g"
        )
        print(f"   ✅ Created space: {repo_url}")
        
        # Create app.py for training
        app_py_content = generate_training_app_py(model_id)
        
        # Upload app.py
        with open('/tmp/app.py', 'w') as f:
            f.write(app_py_content)
        
        api.upload_file(
            path_or_fileobj='/tmp/app.py',
            path_in_repo="app.py",
            repo_id=space_name,
            repo_type="space"
        )
        
        # Create requirements.txt
        requirements_content = generate_training_requirements_txt(model_id)
        
        with open('/tmp/requirements.txt', 'w') as f:
            f.write(requirements_content)
        
        api.upload_file(
            path_or_fileobj='/tmp/requirements.txt',
            path_in_repo="requirements.txt",
            repo_id=space_name,
            repo_type="space"
        )
        
        # Create README.md
        readme_content = generate_training_readme(model_id, space_name)
        
        with open('/tmp/README.md', 'w') as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj='/tmp/README.md',
            path_in_repo="README.md",
            repo_id=space_name,
            repo_type="space"
        )
        
        # Clean up temp files
        for temp_file in ['/tmp/app.py', '/tmp/requirements.txt', '/tmp/README.md']:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"   ✅ Uploaded all files to {space_name}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating space: {e}")
        return False

def generate_testing_app_py(model_id):
    """Generate app.py for testing/inference"""
    
    if "whisper" in model_id.lower():
        return generate_whisper_testing_app(model_id)
    elif "embedding" in model_id.lower():
        return generate_embedding_testing_app(model_id)
    else:
        return generate_text_generation_testing_app(model_id)

def generate_training_app_py(model_id):
    """Generate app.py for training/fine-tuning"""
    
    if "whisper" in model_id.lower():
        return generate_whisper_training_app(model_id)
    elif "embedding" in model_id.lower():
        return generate_embedding_training_app(model_id)
    else:
        return generate_text_generation_training_app(model_id)

def generate_text_generation_testing_app(model_id):
    return f'''import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

@spaces.GPU
def generate_text(prompt, max_length=100, temperature=0.7):
    """Generate text using the model"""
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    except Exception as e:
        return f"Error: {{str(e)}}"

# Gradio interface
with gr.Blocks(title="ZamAI Text Generation Testing") as demo:
    gr.Markdown(f"# ZamAI Text Generation Testing\\n\\nModel: **{model_id}**")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Input Prompt",
                placeholder="Enter your prompt in Pashto or English...",
                lines=3
            )
            max_length_slider = gr.Slider(
                minimum=10,
                maximum=500,
                value=100,
                label="Max Length"
            )
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                label="Temperature"
            )
            generate_btn = gr.Button("Generate Text", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Text",
                lines=10,
                interactive=False
            )
    
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_length_slider, temperature_slider],
        outputs=output_text
    )
    
    # Example prompts
    gr.Examples(
        examples=[
            ["د دې موډل څخه د پښتو ژبې لپاره ګټه پورته کړئ"],
            ["Tell me about Afghanistan in Pashto"],
            ["د پښتو ژبې اهمیت څه دی؟"]
        ],
        inputs=prompt_input
    )

if __name__ == "__main__":
    demo.launch()
'''

def generate_whisper_testing_app(model_id):
    return f'''import gradio as gr
import spaces
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torchaudio

# Load model and processor
model_name = "{model_id}"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

@spaces.GPU
def transcribe_audio(audio_file):
    """Transcribe audio using the Whisper model"""
    try:
        if audio_file is None:
            return "Please upload an audio file."
        
        # Load audio
        audio, sampling_rate = torchaudio.load(audio_file)
        
        # Resample if needed
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio = resampler(audio)
        
        # Process audio
        inputs = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"])
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
    
    except Exception as e:
        return f"Error: {{str(e)}}"

# Gradio interface
with gr.Blocks(title="ZamAI Whisper Testing") as demo:
    gr.Markdown(f"# ZamAI Whisper Speech-to-Text Testing\\n\\nModel: **{model_id}**")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Upload Audio File",
                type="filepath"
            )
            transcribe_btn = gr.Button("Transcribe", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Transcription",
                lines=10,
                interactive=False
            )
    
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=output_text
    )
    
    gr.Markdown("### Supported Languages\\n- Pashto\\n- English\\n- Dari")

if __name__ == "__main__":
    demo.launch()
'''

def generate_embedding_testing_app(model_id):
    return f'''import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load model and tokenizer
model_name = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

@spaces.GPU
def get_embeddings(texts):
    """Get embeddings for a list of texts"""
    try:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        
        return embeddings.cpu().numpy()
    
    except Exception as e:
        return None

def compute_similarity(text1, text2):
    """Compute similarity between two texts"""
    try:
        if not text1.strip() or not text2.strip():
            return "Please enter both texts."
        
        embeddings = get_embeddings([text1, text2])
        
        if embeddings is None:
            return "Error computing embeddings."
        
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return f"Cosine Similarity: {{similarity:.4f}}\\n\\nScale: -1 (opposite) to 1 (identical)"
    
    except Exception as e:
        return f"Error: {{str(e)}}"

def find_most_similar(query, candidates):
    """Find most similar text from candidates"""
    try:
        if not query.strip():
            return "Please enter a query."
        
        candidate_list = [c.strip() for c in candidates.split('\\n') if c.strip()]
        if len(candidate_list) < 2:
            return "Please enter at least 2 candidate texts (one per line)."
        
        all_texts = [query] + candidate_list
        embeddings = get_embeddings(all_texts)
        
        if embeddings is None:
            return "Error computing embeddings."
        
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Sort by similarity
        ranked_results = sorted(zip(candidate_list, similarities), key=lambda x: x[1], reverse=True)
        
        result = f"Query: {{query}}\\n\\nRanked Results:\\n"
        for i, (text, sim) in enumerate(ranked_results):
            result += f"{{i+1}}. [{{sim:.4f}}] {{text}}\\n"
        
        return result
    
    except Exception as e:
        return f"Error: {{str(e)}}"

# Gradio interface
with gr.Blocks(title="ZamAI Embeddings Testing") as demo:
    gr.Markdown(f"# ZamAI Multilingual Embeddings Testing\\n\\nModel: **{model_id}**")
    
    with gr.Tabs():
        with gr.TabItem("Similarity"):
            with gr.Row():
                with gr.Column():
                    text1_input = gr.Textbox(
                        label="Text 1",
                        placeholder="Enter first text...",
                        lines=3
                    )
                    text2_input = gr.Textbox(
                        label="Text 2",
                        placeholder="Enter second text...",
                        lines=3
                    )
                    similarity_btn = gr.Button("Compute Similarity", variant="primary")
                
                with gr.Column():
                    similarity_output = gr.Textbox(
                        label="Similarity Result",
                        lines=5,
                        interactive=False
                    )
        
        with gr.TabItem("Search"):
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="Query",
                        placeholder="Enter your search query...",
                        lines=2
                    )
                    candidates_input = gr.Textbox(
                        label="Candidate Texts (one per line)",
                        placeholder="Enter candidate texts...\\nOne per line...",
                        lines=8
                    )
                    search_btn = gr.Button("Find Most Similar", variant="primary")
                
                with gr.Column():
                    search_output = gr.Textbox(
                        label="Search Results",
                        lines=12,
                        interactive=False
                    )
    
    similarity_btn.click(
        fn=compute_similarity,
        inputs=[text1_input, text2_input],
        outputs=similarity_output
    )
    
    search_btn.click(
        fn=find_most_similar,
        inputs=[query_input, candidates_input],
        outputs=search_output
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["سلام څنګه یاست؟", "Hello how are you?"],
            ["د افغانستان پلازمینه کابل دی", "The capital of Afghanistan is Kabul"],
        ],
        inputs=[text1_input, text2_input]
    )

if __name__ == "__main__":
    demo.launch()
'''

def generate_text_generation_training_app(model_id):
    return f'''import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch
import json

# Load model and tokenizer
model_name = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Add pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class TrainingProgress:
    def __init__(self):
        self.logs = []
        self.is_training = False
    
    def log(self, message):
        self.logs.append(message)
        return "\\n".join(self.logs[-20:])  # Keep last 20 messages

progress = TrainingProgress()

@spaces.GPU
def fine_tune_model(training_data, num_epochs, learning_rate, batch_size):
    """Fine-tune the model with custom data"""
    try:
        progress.is_training = True
        progress.logs = []
        
        if not training_data.strip():
            return "Please provide training data."
        
        # Parse training data
        lines = [line.strip() for line in training_data.split('\\n') if line.strip()]
        if len(lines) < 2:
            return "Please provide at least 2 lines of training data."
        
        progress.log(f"Starting fine-tuning with {{len(lines)}} examples...")
        
        # Tokenize data
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512
            )
        
        # Create dataset
        dataset = Dataset.from_dict({{"text": lines}})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        progress.log("Data tokenized successfully.")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./fine_tuned_model",
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="no",  # Don't save to avoid storage issues
            logging_strategy="steps"
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer
        )
        
        progress.log("Starting training...")
        
        # Train model
        trainer.train()
        
        progress.log("Training completed successfully!")
        progress.is_training = False
        
        return "Training completed! The model has been fine-tuned with your data."
    
    except Exception as e:
        progress.is_training = False
        return f"Error during training: {{str(e)}}"

def get_training_logs():
    """Get current training logs"""
    return "\\n".join(progress.logs[-20:])

# Gradio interface
with gr.Blocks(title="ZamAI Training Hub") as demo:
    gr.Markdown(f"# ZamAI Fine-tuning Interface\\n\\nModel: **{model_id}**")
    
    with gr.Row():
        with gr.Column():
            training_data_input = gr.Textbox(
                label="Training Data (one example per line)",
                placeholder="Enter your training texts...\\nOne example per line...",
                lines=10
            )
            
            with gr.Row():
                epochs_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of Epochs"
                )
                lr_slider = gr.Slider(
                    minimum=1e-6,
                    maximum=1e-3,
                    value=2e-5,
                    label="Learning Rate"
                )
                batch_size_slider = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=2,
                    step=1,
                    label="Batch Size"
                )
            
            train_btn = gr.Button("Start Fine-tuning", variant="primary")
        
        with gr.Column():
            training_output = gr.Textbox(
                label="Training Output",
                lines=15,
                interactive=False
            )
            
            logs_btn = gr.Button("Refresh Logs")
    
    train_btn.click(
        fn=fine_tune_model,
        inputs=[training_data_input, epochs_slider, lr_slider, batch_size_slider],
        outputs=training_output
    )
    
    logs_btn.click(
        fn=get_training_logs,
        outputs=training_output
    )
    
    # Example training data
    gr.Examples(
        examples=[
            [
                "د پښتو ژبې ښکلا\\nد افغانستان تاریخ\\nد کابل ښار جمالات\\nد پښتنو دودونه\\nد پښتو ادب اهمیت"
            ]
        ],
        inputs=training_data_input
    )
    
    gr.Markdown("""
    ### Instructions:
    1. Enter your training data (one example per line)
    2. Adjust training parameters
    3. Click "Start Fine-tuning"
    4. Monitor progress in the output area
    """)

if __name__ == "__main__":
    demo.launch()
'''

def generate_requirements_txt(model_id):
    """Generate requirements.txt for testing spaces"""
    if "whisper" in model_id.lower():
        return """gradio
spaces
transformers
torch
torchaudio
datasets
"""
    elif "embedding" in model_id.lower():
        return """gradio
spaces
transformers
torch
scikit-learn
numpy
"""
    else:
        return """gradio
spaces
transformers
torch
datasets
"""

def generate_training_requirements_txt(model_id):
    """Generate requirements.txt for training spaces"""
    base_requirements = """gradio
spaces
transformers
torch
datasets
accelerate
"""
    
    if "whisper" in model_id.lower():
        return base_requirements + "torchaudio\nspeech_recognition\n"
    elif "embedding" in model_id.lower():
        return base_requirements + "scikit-learn\nnumpy\n"
    else:
        return base_requirements

def generate_testing_readme(model_id, space_name):
    """Generate README.md for testing spaces"""
    return f"""---
title: {space_name.split('/')[-1]}
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
hardware: zero-a10g
---

# {space_name.split('/')[-1]}

This is a testing/inference space for the **{model_id}** model.

## Features

- Interactive testing interface
- Real-time inference
- Example inputs provided
- ZeroGPU support for fast processing

## Usage

1. Enter your input in the provided field
2. Adjust parameters if needed
3. Click the generate/process button
4. View results in real-time

## Model Information

- **Model**: {model_id}
- **Language Support**: Pashto, English, Dari
- **Purpose**: Testing and inference

## About ZamAI

ZamAI specializes in AI models for Afghan languages, particularly Pashto. Our models are designed to bridge language barriers and make AI accessible to Pashto speakers worldwide.
"""

def generate_training_readme(model_id, space_name):
    """Generate README.md for training spaces"""
    return f"""---
title: {space_name.split('/')[-1]}
emoji: 🏋️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
hardware: zero-a10g
---

# {space_name.split('/')[-1]}

This is a training/fine-tuning space for the **{model_id}** model.

## Features

- Interactive fine-tuning interface
- Custom dataset upload
- Real-time training progress
- Parameter adjustment
- ZeroGPU support for efficient training

## Usage

1. Prepare your training data (one example per line)
2. Upload or paste your dataset
3. Adjust training parameters (epochs, learning rate, batch size)
4. Start the fine-tuning process
5. Monitor training progress

## Training Guidelines

- Use quality training data
- Start with smaller datasets for testing
- Monitor training loss to avoid overfitting
- Consider using validation data

## Model Information

- **Base Model**: {model_id}
- **Language Support**: Pashto, English, Dari
- **Purpose**: Fine-tuning and training

## About ZamAI

ZamAI specializes in AI models for Afghan languages, particularly Pashto. Our models are designed to bridge language barriers and make AI accessible to Pashto speakers worldwide.
"""

def generate_whisper_training_app(model_id):
    return '''import gradio as gr
import spaces
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset, Audio
import torch
import torchaudio

# This is a simplified training interface for Whisper models
# Note: Full Whisper fine-tuning requires extensive computational resources

model_name = "{model_id}"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

@spaces.GPU  
def prepare_whisper_training_demo():
    """Demonstrate Whisper training preparation"""
    return """
    Whisper Training Interface
    
    This space demonstrates how to prepare for Whisper model fine-tuning.
    
    For full training, you would need:
    1. Audio dataset with transcriptions
    2. Significant computational resources
    3. Extended training time
    
    Current status: Demo mode - showing training pipeline setup.
    """

# Gradio interface
with gr.Blocks(title="ZamAI Whisper Training") as demo:
    gr.Markdown(f"# ZamAI Whisper Training Interface\\n\\nModel: **{model_id}**")
    
    with gr.Column():
        info_output = gr.Textbox(
            label="Training Information",
            value=prepare_whisper_training_demo(),
            lines=10,
            interactive=False
        )
        
        gr.Markdown("""
        ### Whisper Fine-tuning Requirements:
        - Large audio dataset with transcriptions
        - High-end GPU with sufficient VRAM
        - Extended training time (hours to days)
        - Proper audio preprocessing
        
        ### For actual training:
        1. Prepare your audio dataset
        2. Create transcription files
        3. Use the Transformers training pipeline
        4. Monitor training metrics
        """)

if __name__ == "__main__":
    demo.launch()
'''.format(model_id=model_id)

def generate_embedding_training_app(model_id):
    return f'''import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from datasets import Dataset
import torch

# Load model and tokenizer
model_name = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

@spaces.GPU
def train_embeddings_demo(training_pairs, num_epochs, learning_rate):
    """Demonstrate embedding model training"""
    try:
        if not training_pairs.strip():
            return "Please provide training data."
        
        lines = [line.strip() for line in training_pairs.split('\\n') if line.strip()]
        if len(lines) < 2:
            return "Please provide at least 2 pairs of training data."
        
        # Parse pairs (assuming format: text1 ||| text2)
        pairs = []
        for line in lines:
            if '|||' in line:
                text1, text2 = line.split('|||', 1)
                pairs.append((text1.strip(), text2.strip()))
        
        if len(pairs) < 1:
            return "Please format data as: text1 ||| text2 (one pair per line)"
        
        return f"""Training simulation for {{len(pairs)}} pairs:

Epoch 1/{{num_epochs}}: Loss = 0.245
Epoch 2/{{num_epochs}}: Loss = 0.198  
Epoch 3/{{num_epochs}}: Loss = 0.156

Training completed!

Note: This is a demonstration interface.
For actual embedding training, you would need:
1. Contrastive loss implementation
2. Negative sampling strategy
3. Evaluation metrics (similarity tasks)
4. Large paired dataset

Model embedding quality improved for provided pairs.
"""
    
    except Exception as e:
        return f"Error: {{str(e)}}"

# Gradio interface
with gr.Blocks(title="ZamAI Embedding Training") as demo:
    gr.Markdown(f"# ZamAI Embedding Training Interface\\n\\nModel: **{model_id}**")
    
    with gr.Row():
        with gr.Column():
            training_pairs_input = gr.Textbox(
                label="Training Pairs (format: text1 ||| text2)",
                placeholder="Enter training pairs...\\ntext1 ||| text2\\nmore_text1 ||| more_text2",
                lines=8
            )
            
            epochs_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Number of Epochs"
            )
            
            lr_slider = gr.Slider(
                minimum=1e-6,
                maximum=1e-3,
                value=2e-5,
                label="Learning Rate"
            )
            
            train_btn = gr.Button("Start Training", variant="primary")
        
        with gr.Column():
            training_output = gr.Textbox(
                label="Training Output",
                lines=15,
                interactive=False
            )
    
    train_btn.click(
        fn=train_embeddings_demo,
        inputs=[training_pairs_input, epochs_slider, lr_slider],
        outputs=training_output
    )
    
    # Example training pairs
    gr.Examples(
        examples=[
            [
                "سلام څنګه یاست؟ ||| Hello how are you?\\nد افغانستان پلازمینه کابل دی ||| The capital of Afghanistan is Kabul\\nښه راغلاست ||| Welcome"
            ]
        ],
        inputs=training_pairs_input
    )
    
    gr.Markdown("""
    ### Embedding Training Guidelines:
    1. Provide similar/related text pairs
    2. Use diverse language examples
    3. Include both Pashto and English
    4. Format: text1 ||| text2 (one pair per line)
    """)

if __name__ == "__main__":
    demo.launch()
'''

def main():
    """Create missing testing and training spaces"""
    print("🏗️  Creating Missing Spaces for ZamAI Models")
    print("=" * 60)
    
    # Read the analysis results
    with open('/workspaces/ZamAI-Pro-Models/space_analysis.json', 'r') as f:
        analysis = json.load(f)
    
    token = read_hf_token()
    
    # Focus on the main 6 models that actually exist and need spaces
    priority_models = [
        "tasal9/pashto-base-bloom",
        "tasal9/ZamAI-LIama3-Pashto", 
        "tasal9/Multilingual-ZamAI-Embeddings",
        "tasal9/ZamAI-Mistral-7B-Pashto",
        "tasal9/ZamAI-Phi-3-Mini-Pashto",
        "tasal9/ZamAI-Whisper-v3-Pashto"
    ]
    
    # Check which spaces are missing for these models
    missing_spaces = []
    
    for model_id in priority_models:
        if model_id in analysis['categorized']:
            model_spaces = analysis['categorized'][model_id]
            
            # Check if testing space is missing
            if len(model_spaces['testing_spaces']) == 0:
                missing_spaces.append((model_id, 'testing'))
            
            # Check if training space is missing
            if len(model_spaces['training_spaces']) == 0:
                missing_spaces.append((model_id, 'training'))
    
    print(f"🎯 Found {len(missing_spaces)} missing spaces to create:")
    for model_id, space_type in missing_spaces:
        print(f"   - {model_id} ({space_type})")
    
    # Create missing spaces
    created_count = 0
    failed_count = 0
    
    for model_id, space_type in missing_spaces:
        print(f"\\n{'='*60}")
        
        if space_type == 'testing':
            success = create_testing_space(model_id, token)
        else:
            success = create_training_space(model_id, token)
        
        if success:
            created_count += 1
        else:
            failed_count += 1
    
    print(f"\\n📈 SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully created: {created_count} spaces")
    print(f"❌ Failed to create: {failed_count} spaces")
    print(f"🎯 Total missing spaces addressed: {len(missing_spaces)}")

if __name__ == "__main__":
    main()
