import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import spaces

MODEL_ID = "tasal9/ZamAI-Phi-3-Mini-Pashto"

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
        return f"Error: {e}"

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
        return f"❌ Training error: {e}"

# Gradio Interface
with gr.Blocks(title="ZamAI phi3-pashto Training", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 ZamAI Model Training Space")
    gr.Markdown(f"**Model:** {MODEL_ID}")
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
        - **Model ID:** {MODEL_ID}
        - **Type:** Text Generation
        - **Description:** Pashto Phi-3 model training space
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
