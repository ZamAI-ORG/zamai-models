---
title: ZamAI mT5-Pashto Training
emoji: 🌐
colorFrom: teal
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
---

# 🌐 ZamAI mT5 Pashto Translation + Training Space

Fine-tune and test the `tasal9/ZamAI-mT5-Pashto` translator directly on ZeroGPU.

## Features

- 🔁 **Bidirectional translation demo** (English↔Pashto)
- ⚙️ **LoRA fine-tuning pipeline** for `google/mt5-base`
- 🧹 Automatic dataset column detection (`input/output`, `en/ps`, `prompt/completion`)
- 🚀 Push trained adapters back to Hugging Face Hub in one click
- 📉 Lightweight runs by capping max samples and sequence length

## Usage

1. Open the "Translate" tab to verify the current model.
2. Switch to "Training" and configure:
   - Dataset repo (default: `tasal9/ZamAi-Pashto-Datasets-V2`)
   - Translation direction (English→Pashto or Pashto→English)
   - Epochs, learning rate, and max samples
   - Target repo for uploading LoRA adapters
3. Click **Start Training**. Progress and errors stream into the status box.

## Requirements

Install dependencies listed in `requirements.txt` or let Hugging Face Spaces install them automatically.

## Tips

- Keep `max training samples` low (≤1500) for smoke tests.
- Provide a valid `HF_TOKEN` secret inside the Space to enable automatic pushes.
- Datasets with `instruction/input/output` or `prompt/completion` columns work out-of-the-box.

Happy translating! 🇦🇫