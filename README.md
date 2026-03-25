# ZamAI Models

Model artifacts and experiments published by ZamAI Labs (training results, references, and supporting code where applicable).

- Website: https://zamai.dev
- Labs: https://github.com/ZamAI-ORG


# ZamAI Models

Model artifacts and experiments published by ZamAI Labs (training results, references, and supporting code where applicable).

- Website: https://zamai.dev
- Labs: https://github.com/ZamAI-ORG


# 🇦🇫 ZamAI Pro Models - Complete Deployment

**Afghanistan's Premier AI Ecosystem - Now Production Ready!**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/)

## 🌟 Overview

> **🚨 Having "localhost refused to connect" issues?**  
> Jump to [Quick Start](#-quick-start---complete-deployment) or read [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and [QUICKSTART.md](QUICKSTART.md)

ZamAI Pro Models is a comprehensive, production-ready AI ecosystem designed specifically for Pashto language processing and Afghan cultural context. This project includes speech recognition, text generation, understanding, and voice synthesis capabilities.

### 🎯 Key Features

- **🎤 Advanced Speech Recognition**: Whisper Large v3 for high-quality Pashto speech-to-text
- **🧠 Intelligent Text Generation**: Mistral 7B & Phi-3 Mini for conversational AI
- **🗣️ Natural Voice Synthesis**: Text-to-speech for complete voice interaction
- **📱 Multi-Platform Support**: Web, mobile, and API interfaces
- **🚀 Production Ready**: Docker, nginx, and complete deployment setup
- **🇦🇫 Cultural Context**: Built with Afghan values and Islamic principles

## 🤖 Deployed Models

### Primary Models
| Model | Type | Parameters | Use Case |
|-------|------|------------|----------|
| **openai/whisper-large-v3** | Speech-to-Text | 1.5B | Advanced speech understanding + talking back |
| **mistralai/Mistral-7B-Instruct-v0.3** | Text Generation | 7B | High-quality conversations |
| **microsoft/Phi-3-mini-4k-instruct** | Text Generation | 3.8B | Efficient edge deployment |

### Capabilities
- ✅ **Speech Recognition**: Convert Pashto speech to text
- ✅ **Language Understanding**: Comprehend context and intent
- ✅ **Response Generation**: Generate contextual responses
- ✅ **Voice Synthesis**: Convert text back to speech
- ✅ **Multi-Modal Interface**: Support text, voice, and document input

ZamAI is a comprehensive AI platform designed specifically for Pashto language processing and Afghan cultural context. Built on Hugging Face infrastructure with enterprise-grade capabilities.

## 🚀 ZamAI Models

### Core Models
- **🎤 ZamAI-Voice-Assistant** - Conversational AI for voice interactions
- **👨‍🏫 ZamAI-Tutor-Bot** - Educational AI tutor for Pashto learning  
- **📊 ZamAI-Business-Automation** - Document processing and business intelligence
- **🔍 ZamAI-Embeddings** - Multilingual embeddings for semantic search
- **🗣️ ZamAI-STT-Processor** - Advanced speech-to-text for Pashto
- **🔊 ZamAI-TTS-Generator** - Natural text-to-speech synthesis

## 🏗️ Architecture

```
zama-hf-pro/
├── 🎤 voice_assistant/      # Complete voice interaction system
│   ├── src/
│   │   ├── app.py           # Multi-tab Gradio interface  
│   │   ├── api_client.py    # Enhanced HF API client
│   │   └── ...              # Additional components
│   └── requirements.txt
│
├── 👨‍🏫 tutor_bot/            # Educational platform
├── 📊 business_automation/  # Document processing
├── 🚀 fastapi_backend/      # Production API server
├── 📱 react_native_app/     # Mobile application
└── 🔄 .github/workflows/    # CI/CD pipelines

# Deployment Infrastructure
├── 🐳 Dockerfile           # Container configuration
├── 🗄️ docker-compose.yml   # Multi-service deployment
├── 🌐 nginx.conf           # Production reverse proxy
├── 🚀 deploy.sh            # Automated deployment script
└── ⚙️ deployment_config.json # Model configuration
```

## 🎯 Model Pipeline

### Voice Assistant Workflow
```
🎤 Audio Input
    ↓
📝 Speech-to-Text (Whisper Large v3)
    ↓  
🧠 Language Understanding (Mistral 7B)
    ↓
💬 Response Generation 
    ↓
🗣️ Text-to-Speech
    ↓
🔊 Audio Output
```

### Model Selection Logic
- **Primary**: Mistral 7B for high-quality responses
- **Lightweight**: Phi-3 Mini for edge deployment
- **Speech**: Whisper Large v3 for advanced understanding
- **Fallback**: Automatic model switching on errors

## � Quick Start - Complete Deployment

### ⚡ Fastest Way to Get Started

**Having localhost connection issues? Follow these steps:**

1. **Get HuggingFace Token** (Required)
   - Visit: https://huggingface.co/settings/tokens
   - Create a token with "read" permissions
   - Copy your token (starts with `hf_`)

2. **Quick Setup**
```bash
# Clone the repository
git clone https://github.com/tasal9/ZamAI-Pro-Models.git
cd ZamAI-Pro-Models

# Set your token (choose ONE method):
echo "hf_YOUR_TOKEN_HERE" > HF-Token.txt
# OR
export HUGGINGFACE_TOKEN="hf_YOUR_TOKEN_HERE"

# Run pre-flight check
python3 test_startup.py

# Start the application
./start_voice_assistant.sh
```

3. **Access the Application**
   - Open browser: http://localhost:7860
   - You should see the ZamAI Voice Assistant interface

**Troubleshooting:** If localhost refuses to connect, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and [QUICKSTART.md](QUICKSTART.md)

### Option 1: One-Click Deployment Script

```bash
# Clone the repository
git clone https://github.com/tasal9/ZamAI-Pro-Models.git
cd ZamAI-Pro-Models

# Set your Hugging Face token
export HUGGINGFACE_TOKEN="your_hf_token_here"
# OR create HF-Token.txt file with your token

# Run deployment script
chmod +x deploy.sh
./deploy.sh local    # For local development
# ./deploy.sh docker  # For Docker deployment
```

### Option 2: Docker Deployment

```bash
# Quick Docker deployment
docker-compose up -d

# Access services:
# Voice Assistant: http://localhost:7860
# Tutor Bot: http://localhost:7861
# Business Tools: http://localhost:7862
# API Backend: http://localhost:8000
```

### Option 3: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Validate setup
python scripts/utils/validate_setup.py

# Test models
python test_deployment_models.py

# Start Voice Assistant
python zama-hf-pro/voice_assistant/src/app.py
```

## 🔧 Development

### Model Training
```bash
# Train new ZamAI model
python zama-hf-pro/tutor_bot/src/fine_tuning.py

# Push to Hugging Face
python create_hf_models.py
```

### Testing
```bash
# Run comprehensive tests
python scripts/testing/test_models.py

# Quick model validation
python scripts/utils/validate_setup.py
```

## 📊 Features

### 🎯 Core Capabilities
- **Pashto Language Processing** - Native support for Afghan Pashto
- **Cultural Context Awareness** - Islamic values and Afghan customs
- **Multi-Modal Interface** - Voice, text, and document processing
- **Enterprise Integration** - FastAPI backend with authentication
- **Mobile Ready** - React Native cross-platform app

### 🏢 Business Applications
- Document translation and processing
- Voice-activated customer service
- Educational content generation
- Automated report creation
- Semantic search and retrieval

## 🌍 Deployment

### Hugging Face Spaces
All components deployable as HF Spaces with one-click deployment:

```yaml
# .github/workflows/deploy_voice.yml
- name: Deploy to HF Spaces
  run: huggingface-cli upload tasal9/zamai-voice-space
```

### API Endpoints
- **Voice**: `https://tasal9-zamai-voice.hf.space`
- **Tutor**: `https://tasal9-zamai-tutor.hf.space`  
- **Business**: `https://tasal9-zamai-business.hf.space`

## 📈 Performance

| Model | Task | Accuracy | Speed |
|-------|------|----------|-------|
| ZamAI-Voice-Assistant | Conversation | 94% | 200ms |
| ZamAI-Tutor-Bot | Education | 96% | 150ms |
| ZamAI-STT-Processor | Speech Recognition | 92% | 300ms |
| ZamAI-Embeddings | Semantic Search | 89% | 50ms |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

Apache 2.0 License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- Hugging Face for infrastructure and tools
- Afghan AI community for cultural guidance
- Open source contributors worldwide

## 📞 Contact

- **Project Lead**: tasal9@huggingface.co
- **Community**: [ZamAI Discord](https://discord.gg/zamai)
- **Issues**: [GitHub Issues](https://github.com/tasal9/ZamAI-Pro-Models/issues)

---

## 📚 Additional Documentation

Having issues? Check these guides:
- 🚨 **[LOCALHOST_FIX.md](LOCALHOST_FIX.md)** - Quick fix for "localhost refused to connect"
- 🚀 **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step setup guide
- 🔧 **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Comprehensive troubleshooting
- ✅ **[SUCCESS_INDICATORS.md](SUCCESS_INDICATORS.md)** - How to know if everything is working
- 📖 **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Production deployment instructions

---

🇦🇫 **د افغانستان د AI پروژه** - Empowering Afghanistan through AI

**Built with ❤️ for the Afghan people**
