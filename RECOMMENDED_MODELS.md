# 🎯 Recommended Hugging Face Models for Pashto Fine-tuning

## 🏆 Top Priority Models to Fine-tune

### 1. **For Advanced Chat & Conversation**
```
Model: microsoft/DialoGPT-medium
Purpose: Conversational AI
Fine-tune for: Pashto conversations with Afghan cultural context
Expected output: tasal9/ZamAI-DialoGPT-Pashto-V3
```

### 2. **For High-Quality Text Generation**
```
Model: microsoft/DialoGPT-large
Purpose: Advanced dialogue generation
Fine-tune for: Long-form Pashto conversations
Expected output: tasal9/ZamAI-DialoGPT-Large-Pashto
```

### 3. **For Translation (Pashto ↔ English)**
```
Model: Helsinki-NLP/opus-mt-en-mul
Purpose: Machine translation
Fine-tune for: High-quality Pashto-English translation
Expected output: tasal9/ZamAI-Translator-Pashto-EN
```

### 4. **For Question Answering**
```
Model: deepset/roberta-base-squad2
Purpose: Question answering
Fine-tune for: Pashto Q&A with Afghan knowledge
Expected output: tasal9/ZamAI-QA-Pashto
```

### 5. **For Sentiment Analysis**
```
Model: cardiffnlp/twitter-roberta-base-sentiment-latest
Purpose: Sentiment analysis
Fine-tune for: Pashto emotion and sentiment detection
Expected output: tasal9/ZamAI-Sentiment-Pashto
```

---

## 🚀 Next Generation Models (Advanced)

### 6. **Code-Switching Model**
```
Base: google/mt5-base
Purpose: Handle Pashto-English mixed text
Fine-tune for: Code-switching conversations
Expected output: tasal9/ZamAI-CodeSwitch-Pashto-EN
```

### 7. **Poetry & Literature**
```
Base: gpt2-medium
Purpose: Pashto poetry and literature generation
Fine-tune for: Classical Pashto poetry and ghazals
Expected output: tasal9/ZamAI-Poetry-Pashto
```

### 8. **Islamic Content**
```
Base: UBC-NLP/MARBERTv2
Purpose: Islamic and religious content
Fine-tune for: Islamic teachings in Pashto
Expected output: tasal9/ZamAI-Islamic-Pashto
```

---

## 📊 Your Current vs Recommended Models

| Current Models | Status | Recommended Upgrade |
|----------------|--------|-------------------|
| ✅ ZamAI-Mistral-7B-Pashto | Active | 🔥 Keep & improve |
| ✅ ZamAI-LIama3-Pashto | Active | 🔥 Keep & improve |
| ✅ pashto-base-bloom | Active | ➡️ Upgrade to DialoGPT |
| ✅ Multilingual-ZamAI-Embeddings | Active | 🔥 Keep & improve |
| ❌ Translation Model | Missing | 🆕 Create ZamAI-Translator |
| ❌ Q&A Model | Missing | 🆕 Create ZamAI-QA |
| ❌ Sentiment Model | Missing | 🆕 Create ZamAI-Sentiment |

---

## 🎯 Fine-tuning Priority Order

### Phase 1: Essential Models (Next 2 weeks)
1. **ZamAI-Translator-Pashto-EN** (Translation)
2. **ZamAI-QA-Pashto** (Question Answering)
3. **ZamAI-Sentiment-Pashto** (Sentiment Analysis)

### Phase 2: Advanced Models (Next month)
4. **ZamAI-DialoGPT-Pashto-V3** (Better conversations)
5. **ZamAI-CodeSwitch-Pashto-EN** (Mixed language)
6. **ZamAI-Poetry-Pashto** (Literature)

### Phase 3: Specialized Models (Future)
7. **ZamAI-Islamic-Pashto** (Religious content)
8. **ZamAI-News-Pashto** (News generation)
9. **ZamAI-Education-Pashto** (Educational content)

---

## 🛠️ Fine-tuning Configurations

### For Translation Model:
```json
{
  "base_model": "Helsinki-NLP/opus-mt-en-mul",
  "task": "translation",
  "dataset": "pashto_english_parallel_corpus",
  "epochs": 5,
  "learning_rate": 3e-5,
  "batch_size": 8,
  "max_source_length": 128,
  "max_target_length": 128
}
```

### For Q&A Model:
```json
{
  "base_model": "deepset/roberta-base-squad2",
  "task": "question-answering",
  "dataset": "pashto_qa_dataset",
  "epochs": 3,
  "learning_rate": 2e-5,
  "batch_size": 16,
  "max_seq_length": 384
}
```

### For Sentiment Model:
```json
{
  "base_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
  "task": "text-classification",
  "dataset": "pashto_sentiment_dataset",
  "epochs": 4,
  "learning_rate": 2e-5,
  "batch_size": 32,
  "num_labels": 3
}
```

---

## 📈 Expected Performance Improvements

| Model Type | Current Status | After Fine-tuning |
|------------|---------------|-------------------|
| Chat Quality | Good (70%) | Excellent (90%) |
| Translation | Basic (60%) | Professional (85%) |
| Q&A Accuracy | None | Good (80%) |
| Sentiment | None | Good (75%) |
| Cultural Context | Good (65%) | Excellent (90%) |

---

## 🎯 Immediate Action Plan

### This Week:
1. **Create fine-tuning scripts** for translation model
2. **Prepare Pashto-English parallel dataset**
3. **Start fine-tuning translation model**

### Next Week:
1. **Create Q&A dataset** with Afghan knowledge
2. **Fine-tune Q&A model**
3. **Test translation model**

### Following Week:
1. **Create sentiment dataset**
2. **Fine-tune sentiment model**
3. **Integrate all models into ZamAI V3**

---

## 💡 Dataset Requirements

### Translation Dataset:
- **Size:** 50K+ Pashto-English pairs
- **Sources:** News, literature, conversations
- **Quality:** Human-validated translations

### Q&A Dataset:
- **Size:** 10K+ Pashto Q&A pairs
- **Topics:** Afghan history, culture, Islam, general knowledge
- **Format:** Question + Context + Answer

### Sentiment Dataset:
- **Size:** 20K+ labeled Pashto texts
- **Labels:** Positive, Negative, Neutral
- **Sources:** Social media, reviews, comments

🎯 **Goal:** Make ZamAI V3 the most comprehensive Pashto AI assistant!
