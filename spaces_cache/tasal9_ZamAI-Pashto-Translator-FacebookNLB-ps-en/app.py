# fine_tune_facebooknlb_pashto.py

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
import spaces

# --------------------------
# CONFIG
# --------------------------
MODEL_NAME = "tasal9/ZamAI-Pashto-Translator-FacebookNLB-ps-en"
DATASET_NAME = "tasal9/ZamAI-Pashto-Mega-Dataset"
OUTPUT_DIR = "./facebooknlb-pashto-finetuned"
PUSH_TO_HF = True  # Set to True to push to Hugging Face Hub

# --------------------------
# LOAD MODEL & TOKENIZER
# --------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# --------------------------
# LOAD DATASET
# --------------------------
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

# Expected dataset format: instruction/input/response OR source/target
# Adjust mapping function based on actual columns in your dataset
@spaces.GPU
def preprocess(examples):
    if "instruction" in examples and "response" in examples:
        inputs = examples["instruction"]
        targets = examples["response"]
    elif "source" in examples and "target" in examples:
        inputs = examples["source"]
        targets = examples["target"]
    else:
        raise ValueError("Dataset does not contain expected columns.")

    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# --------------------------
# TRAINING CONFIG
# --------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
    push_to_hub=PUSH_TO_HF,
    hub_model_id=f"{MODEL_NAME}-finetuned" if PUSH_TO_HF else None,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=20,
    save_strategy="epoch"
)

# --------------------------
# TRAINER
# --------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("validation", tokenized_dataset["train"].select(range(200))),
    tokenizer=tokenizer,
    data_collator=data_collator
)

# --------------------------
# START TRAINING
# --------------------------
print("Starting training...")
trainer.train()

# --------------------------
# SAVE MODEL
# --------------------------
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

if PUSH_TO_HF:
    trainer.push_to_hub()
    tokenizer.push_to_hub(f"{MODEL_NAME}-finetuned")

print("✅ Fine-tuning complete!")