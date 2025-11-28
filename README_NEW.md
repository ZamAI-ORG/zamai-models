# 🇦🇫 ZamAI Hugging Face Models

A comprehensive toolkit for training, testing, and managing Pashto language AI models using Hugging Face.

## 📁 Project Structure

```
huggingface-models/
├── 📚 Documentation/
│   ├── README.md                    # This file
│   ├── CURRENT_MODELS.md           # Current model inventory
│   ├── RECOMMENDED_MODELS.md       # Future model recommendations
│   ├── MODEL_CATALOG.md            # Complete model catalog
│   └── MODEL_FIXES.md             # Model fixes and updates
│
├── 🛠️ Scripts/
│   ├── training/                   # Training scripts
│   │   ├── train_zamai_v4.py      # Main ZamAI V4 training
│   │   ├── train_pashto_chat.py   # Pashto chat model training
│   │   └── training_pipeline.py   # Automated training pipeline
│   │
│   ├── testing/                    # Model testing scripts
│   │   ├── test_models.py         # Test all models
│   │   ├── check_hf_models.py     # Check HF model status
│   │   ├── quick_model_test.py    # Quick model testing
│   │   └── simple_test.py         # Simple model test
│   │
│   ├── analysis/                   # Data and model analysis
│   │   ├── analyze_zamai_dataset.py # Dataset analysis
│   │   ├── explore_dataset.py     # Dataset exploration
│   │   ├── explore_zamai_dataset.py # ZamAI dataset explorer
│   │   └── debug_models.py        # Model debugging
│   │
│   └── utils/                      # Utility scripts
│       ├── validate_setup.py      # Setup validation
│       ├── basic_check.py         # Basic environment check
│       ├── check_dataset_access.py # Dataset access check
│       ├── model_manager.py       # Model management
│       └── quick_dataset_check*.py # Quick dataset checks
│
├── ⚙️ Configs/
│   └── pashto_chat_config.json    # Main training configuration
│
├── 🗂️ Models/
│   ├── text-generation/           # Text generation models
│   ├── embeddings/                # Embedding models
│   ├── chat/                      # Chat models
│   └── translation/               # Translation models
│
├── 💾 Data/
│   ├── raw/                       # Raw datasets
│   └── processed/                 # Processed data and results
│
├── 📊 Outputs/
│   └── (Training outputs will be saved here)
│
├── 📝 Logs/
│   └── (Training logs will be saved here)
│
└── 🔑 HF-Token.txt               # Hugging Face authentication token
```

## 🚀 Quick Start

### 1. Validate Setup
```bash
python scripts/utils/validate_setup.py
```

### 2. Analyze Your Dataset
```bash
python scripts/analysis/analyze_zamai_dataset.py
```

### 3. Test Existing Models
```bash
python scripts/testing/test_models.py
```

### 4. Train New Model
```bash
python scripts/training/train_zamai_v4.py
```

## 📊 Your Current Models

- **tasal9/pashto-base-bloom** - BLOOM-based Pashto model
- **tasal9/pashto-bloom-base** - Base BLOOM for Pashto
- **tasal9/ZamAI-LIama3-Pashto** - Llama3-based ZamAI
- **tasal9/Multilingual-ZamAI-Embeddings** - Multilingual embeddings
- **tasal9/ZamAI-Mistral-7B-Pashto** - Mistral-based ZamAI

## 🗃️ Your Dataset

- **tasal9/ZamAI_Pashto_Dataset** - Instruction-following Pashto dataset
- Format: `{instruction, input, output}`
- Files: `pashto_train_instruction.jsonl`, `pashto_val_instruction.jsonl`
- **Normalize prompt/completion files**: run `python scripts/datasets/normalize_prompt_completion.py --source <prompt_file> --destination <instruction_file>` before uploading any `pashto_*_prompt_completion.jsonl` splits to Hugging Face to avoid schema mismatches.

## ⚙️ Configuration

Main configuration file: `configs/pashto_chat_config.json`

Key settings:
- **Base Model**: Llama-3.1-8B-Instruct
- **Dataset**: tasal9/ZamAI_Pashto_Dataset
- **Output**: tasal9/zamai-pashto-chat-8b
- **LoRA**: Efficient fine-tuning enabled

## 🎯 Training Pipeline

1. **Dataset Loading**: Loads your instruction-following dataset
2. **Model Preparation**: Downloads and prepares base model
3. **LoRA Setup**: Configures efficient fine-tuning
4. **Training**: Trains with your Pashto data
5. **Evaluation**: Validates on held-out data
6. **Hub Upload**: Automatically pushes to Hugging Face

## 📈 Monitoring

- **Weights & Biases**: Enabled for training monitoring
- **Logging**: Comprehensive logging to `logs/` directory
- **Checkpoints**: Regular model checkpoints saved

## 🛠️ Development

### Adding New Models
1. Add model config to `models/` directory
2. Update `CURRENT_MODELS.md`
3. Run training with new config

### Testing Models
- Use `scripts/testing/test_models.py` for comprehensive testing
- Quick tests available with `scripts/testing/quick_model_test.py`

### Dataset Analysis
- Run `scripts/analysis/analyze_zamai_dataset.py` for detailed analysis
- Explore data with `scripts/analysis/explore_zamai_dataset.py`

## 🔧 Troubleshooting

1. **Setup Issues**: Run `scripts/utils/validate_setup.py`
2. **Model Errors**: Check `scripts/analysis/debug_models.py`
3. **Dataset Problems**: Use `scripts/utils/check_dataset_access.py`

## 📚 Documentation

- `CURRENT_MODELS.md` - Your current model inventory
- `RECOMMENDED_MODELS.md` - Future model recommendations
- `MODEL_CATALOG.md` - Complete model catalog with details
- `MODEL_FIXES.md` - Known issues and fixes

---

🇦🇫 **ZamAI Project** - Advancing Pashto Language AI
