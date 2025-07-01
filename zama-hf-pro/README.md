# 🇦🇫 ZamAI x Hugging Face Pro

Advanced Pashto AI solutions powered by Hugging Face.

## 🚀 Components

### 🎤 Voice Assistant
Gradio-based voice interface with STT→LLM→TTS pipeline.
```bash
cd voice_assistant
pip install -r requirements.txt
python src/app.py
```

### 📚 Tutor Bot
Educational chatbot with fine-tuned Pashto models.
```bash
cd tutor_bot
pip install -r requirements.txt
python src/app.py
```

### 💼 Business Automation
Document processing and analysis tools.
```bash
cd business_automation
pip install -r requirements.txt
python src/app.py
```

### 🚀 FastAPI Backend
REST API for all ZamAI services.
```bash
cd fastapi_backend
pip install -r requirements.txt
python src/main.py
```

### 📱 React Native App
Mobile application for ZamAI services.
```bash
cd react_native_app
npm install
npm start
```

## 🔑 Setup

1. Set HF_TOKEN in your environment
2. Configure GitHub secrets for CI/CD
3. Deploy to Hugging Face Spaces

## 🇦🇫 Models

- tasal9/ZamAI-Mistral-7B-Pashto
- tasal9/ZamAI-LIama3-Pashto  
- tasal9/pashto-tutor-bot
- tasal9/Multilingual-ZamAI-Embeddings

Built with ❤️ for Afghanistan 🇦🇫
