# ZamAI Voice Assistant

An advanced Pashto language voice interface powered by Hugging Face Pro.

## 🌟 Features

- **High-quality Pashto STT**: Advanced speech recognition optimized for Pashto dialects
- **Cultural Context-Aware NLU**: Natural language understanding with Afghan cultural sensitivity
- **Natural Pashto TTS**: High-quality speech synthesis with multiple Afghan regional voices
- **Conversation Memory**: Maintains context across the conversation
- **Pro API Integration**: Uses Hugging Face Pro endpoints for higher performance

## 📄 Requirements

```
gradio>=4.0.0
torch>=2.0.0
transformers>=4.30.0
huggingface_hub>=0.16.0
httpx>=0.24.0
numpy>=1.24.0
torchaudio>=2.0.0
soundfile>=0.12.0
requests>=2.28.0
```

## 🚀 Quick Start

1. Set up environment variables:
   ```bash
   export HF_TOKEN="your_hugging_face_token"
   export PRO_ENDPOINT="your_pro_endpoint_url"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python src/app.py
   ```

## 🛠️ Component Details

### Speech-to-Text (STT)
- Uses `tasal9/Pashto-Whisper-Large-Pro` model
- Optimized for various Pashto dialects (Afghanistan and Pakistan)
- Enhanced noise reduction and context-awareness

### Natural Language Understanding (NLU)
- Uses `tasal9/ZamAI-Mistral-7B-Pashto` model
- Cultural sensitivity for Afghan context
- Intent classification and entity recognition

### Text-to-Speech (TTS)
- Uses `tasal9/Pashto-XTTS-Pro` model
- Multiple voice profiles (male/female, regional accents)
- Natural prosody and intonation for Pashto

## 🧩 Integration

This component can be integrated with other ZamAI components:
- **Tutor Bot**: For educational spoken interactions
- **Business Automation**: For voice-driven business workflows
- **React Native App**: For mobile voice interface

## 📊 Monitoring

The system logs performance metrics to the console, including:
- Speech recognition accuracy
- Response generation time
- Voice synthesis quality

## 🔒 Security

- All API calls use authentication with your Hugging Face token
- No user data is stored beyond the current session
- Pro endpoint ensures dedicated resources and better security

## 🔧 Troubleshooting

- **Recognition Issues**: Try adjusting microphone settings or speaking more clearly
- **Slow Responses**: Check your internet connection and Pro endpoint status
- **TTS Problems**: Ensure your audio output is correctly configured

## 📝 License

Copyright (c) 2025 ZamAI - All rights reserved
