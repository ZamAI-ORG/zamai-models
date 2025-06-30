# 🎯 Training Scripts

This directory contains all training scripts for ZamAI models.

## Scripts

### `train_zamai_v4.py`

Main training script for ZamAI V4 using your Pashto dataset.

- **Base Model**: Mistral-7B-Instruct
- **Dataset**: tasal9/ZamAI_Pashto_Dataset
- **Output**: tasal9/ZamAI-V4-Mistral-7B-Pashto

**Usage:**

```bash
python train_zamai_v4.py
```

### `train_pashto_chat.py`

Core training implementation with LoRA fine-tuning.

- Supports instruction-following format
- Memory-efficient training
- Automatic model upload to HF Hub

**Usage:**

```bash
python train_pashto_chat.py
```

### `training_pipeline.py`

Automated training pipeline for batch processing.

- Multiple model training
- Experiment tracking
- Result comparison

## Configuration

Training configurations are stored in `../configs/pashto_chat_config.json`.

## Output

Trained models are saved to `../outputs/` and automatically uploaded to Hugging Face Hub.
