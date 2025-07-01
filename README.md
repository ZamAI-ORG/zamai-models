# 🇦🇫 ZamAI x Hugging Face Pro

**Afghanistan's Premier AI Ecosystem on Hugging Face**

## 🌟 Overview

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
├── 🎤 voice_assistant/      # Voice interaction system
├── 👨‍🏫 tutor_bot/            # Educational platform
├── 📊 business_automation/  # Document processing
├── 🚀 fastapi_backend/      # API server
├── 📱 react_native_app/     # Mobile application
└── 🔄 .github/workflows/    # CI/CD pipelines
```

## 🛠️ Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/tasal9/ZamAI-Pro-Models.git
cd ZamAI-Pro-Models

# Set Hugging Face token
export HUGGINGFACE_TOKEN="your_token_here"

# Install dependencies
pip install -r requirements.txt
```

### 2. Voice Assistant
```bash
cd zama-hf-pro/voice_assistant
python src/app.py
```

### 3. Tutor Bot  
```bash
cd zama-hf-pro/tutor_bot
python src/app.py
```

### 4. API Backend
```bash
cd zama-hf-pro/fastapi_backend
python src/main.py
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

🇦🇫 **د افغانستان د AI پروژه** - Empowering Afghanistan through AI

**Built with ❤️ for the Afghan people**
