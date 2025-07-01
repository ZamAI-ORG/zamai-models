# 🤗 Pashto Hugging Face Models Catalog

## 📋 Recommended Base Models for Pashto Fine-tuning

### 1. Language Models (for Chat & Text Generation)

#### **Primary Recommendations**

| Model | Hub ID | Parameters | Use Case | Priority |
|-------|--------|------------|----------|----------|
| **Llama 3.1 8B Instruct** | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Pashto Chat, Q&A | ⭐⭐⭐ |
| **Mistral 7B Instruct** | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Pashto Conversations | ⭐⭐⭐ |
| **Qwen2.5 7B Instruct** | `Qwen/Qwen2.5-7B-Instruct` | 7B | Multilingual Support | ⭐⭐ |
| **Gemma 2 9B** | `google/gemma-2-9b-it` | 9B | Instruction Following | ⭐⭐ |

#### **Lightweight Options**

| Model | Hub ID | Parameters | Use Case | Priority |
|-------|--------|------------|----------|----------|
| **ZamAI Phi-3 Mini Pashto** | `tasal9/ZamAI-Phi-3-Mini-Pashto` | 2.7B | Pashto Edge Deployment | ⭐⭐⭐ |
| **Phi-3 Mini 4K** | `microsoft/Phi-3-mini-4k-instruct` | 2.7B | Edge Deployment | ⭐⭐ |
| **Phi-3.5 Mini** | `microsoft/Phi-3.5-mini-instruct` | 3.8B | Edge Deployment | ⭐⭐ |
| **Gemma 2 2B** | `google/gemma-2-2b-it` | 2B | Mobile/Local | ⭐ |

### 2. Translation Models

#### **Multilingual Translation**

| Model | Hub ID | Languages | Use Case | Priority |
|-------|--------|-----------|----------|----------|
| **NLLB-200** | `facebook/nllb-200-3.3B` | 200+ langs | Pashto↔English | ⭐⭐⭐ |
| **M2M-100** | `facebook/m2m100_1.2B` | 100 langs | Multi-direction | ⭐⭐ |
| **mT5-Large** | `google/mt5-large` | 101 langs | Custom Translation | ⭐⭐ |

### 3. Text Classification & Analysis

#### **Sentiment & Classification**

| Model | Hub ID | Use Case | Priority |
|-------|--------|----------|----------|
| **XLM-RoBERTa Large** | `xlm-roberta-large` | Sentiment Analysis | ⭐⭐⭐ |
| **mBERT** | `bert-base-multilingual-cased` | Text Classification | ⭐⭐ |
| **DistilBERT Multilingual** | `distilbert-base-multilingual-cased` | Lightweight Classification | ⭐⭐ |

### 4. Speech Models

#### **Speech Recognition (ASR)**

| Model | Hub ID | Use Case | Priority |
|-------|--------|----------|----------|
| **ZamAI Whisper v3 Pashto** | `tasal9/ZamAI-Whisper-v3-Pashto` | Pashto-optimized Speech→Text | ⭐⭐⭐ |
| **Whisper Large v3** | `openai/whisper-large-v3` | General Speech→Text + Understanding | ⭐⭐⭐ |
| **Wav2Vec2 XLS-R** | `facebook/wav2vec2-xls-r-300m` | Multilingual ASR | ⭐⭐ |
| **SpeechT5** | `microsoft/speecht5_asr` | Speech Recognition | ⭐ |

#### **Text-to-Speech (TTS)**

| Model | Hub ID | Use Case | Priority |
|-------|--------|----------|----------|
| **SpeechT5 TTS** | `microsoft/speecht5_tts` | Text→Speech | ⭐⭐ |
| **VITS** | `facebook/mms-tts` | Multilingual TTS | ⭐⭐ |

### 5. Embedding Models

#### **Semantic Embeddings**

| Model | Hub ID | Use Case | Priority |
|-------|--------|----------|----------|
| **Sentence-BERT Multilingual** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Semantic Search | ⭐⭐⭐ |
| **LaBSE** | `sentence-transformers/LaBSE` | Cross-lingual Embeddings | ⭐⭐ |
| **Universal Sentence Encoder** | `sentence-transformers/distiluse-base-multilingual-cased-v2` | General Embeddings | ⭐⭐ |

---

## 🎯 Custom Models to Create

### 1. ZamAI Pashto Chat Model
- **Base**: `meta-llama/Llama-3.1-8B-Instruct`
- **Purpose**: Pashto conversational AI
- **Fine-tuning Data**: Pashto conversations, cultural context
- **Target Hub ID**: `tasal9/zamai-pashto-chat-7b`

### 2. ZamAI Pashto Translation
- **Base**: `facebook/nllb-200-3.3B`
- **Purpose**: High-quality Pashto↔English translation
- **Fine-tuning Data**: Pashto-English parallel corpus
- **Target Hub ID**: `tasal9/zamai-pashto-translation`

### 3. ZamAI Pashto Grammar Checker
- **Base**: `xlm-roberta-large`
- **Purpose**: Grammar error detection and correction
- **Fine-tuning Data**: Pashto grammar examples
- **Target Hub ID**: `tasal9/zamai-pashto-grammar`

### 4. ZamAI Pashto Poetry Analyzer
- **Base**: `bert-base-multilingual-cased`
- **Purpose**: Classical Pashto poetry analysis
- **Fine-tuning Data**: Pashto poetry corpus
- **Target Hub ID**: `tasal9/zamai-pashto-poetry`

### 5. ZamAI Pashto Speech Recognition
- **Base**: `openai/whisper-large-v3`
- **Purpose**: Pashto speech recognition
- **Fine-tuning Data**: Pashto audio datasets
- **Target Hub ID**: `tasal9/ZamAI-Whisper-v3-Pashto`
- **Status**: ✅ Deployed

---

## 📊 Model Recommendations by Use Case

### High Priority (Implement First)
1. **Chat**: Llama 3.1 8B + fine-tuning
2. **Translation**: NLLB-200 3.3B + fine-tuning
3. **Speech**: Whisper Large v3 + fine-tuning

### Medium Priority
1. **Grammar**: XLM-RoBERTa + custom training
2. **Embeddings**: Sentence-BERT multilingual
3. **Classification**: mBERT + fine-tuning

### Future Enhancement
1. **Poetry**: Custom BERT + poetry corpus
2. **TTS**: SpeechT5 + Pashto voice data
3. **Summarization**: mT5 + Pashto summaries

---

## 🔧 Implementation Strategy

### Phase 1: Base Model Integration (Week 1-2)
- Set up Hugging Face Hub integration
- Download and test base models
- Create inference pipelines

### Phase 2: Fine-tuning Setup (Week 3-4)
- Prepare Pashto datasets
- Set up fine-tuning scripts
- Train initial custom models

### Phase 3: Deployment (Week 5-6)
- Deploy models to production
- Optimize for performance
- A/B test model performance

### Phase 4: Advanced Models (Week 7-8)
- Create specialized models
- Multi-model ensembles
- Edge deployment optimization

---

## 💾 Storage Requirements

| Model Type | Size | Storage Needed |
|------------|------|----------------|
| 7B-8B Models | ~15GB each | 45GB |
| 3B Models | ~6GB each | 18GB |
| BERT Models | ~1GB each | 5GB |
| Speech Models | ~3GB each | 9GB |
| **Total Estimated** | | **~75GB** |

---

## 🚀 Next Steps

1. **Set up Hugging Face credentials**
2. **Create model download scripts**
3. **Prepare fine-tuning datasets**
4. **Set up training infrastructure**
5. **Deploy to ZamAI Hub**
