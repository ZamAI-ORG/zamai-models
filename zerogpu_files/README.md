---
title: ZamAI Pashto Chat Training
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
---

# 🚀 ZamAI Pashto Chat Model Training

This space trains a Pashto conversational AI model using ZeroGPU infrastructure.

## Features
- ✅ Automatic model setup and LoRA fine-tuning
- ✅ Memory-optimized training with 4-bit quantization
- ✅ Real-time progress monitoring
- ✅ Automatic model upload to HuggingFace Hub
- ✅ Optional WandB experiment tracking

## Usage
1. Click "Setup Model" to prepare the base model
2. Optionally add your WandB API key for tracking
3. Click "Start Training" to begin fine-tuning
4. Wait 2-4 hours for training to complete
5. Your model will be available at `tasal9/zamai-pashto-chat-8b`

Training uses Meta's Llama 3.1 8B as the base model and fine-tunes it on Pashto conversational data using LoRA adapters.
