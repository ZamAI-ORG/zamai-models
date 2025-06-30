# 🌐 Translation Models for Pashto

## Primary Translation Models

### 1. NLLB-200 (No Language Left Behind)
```yaml
model_id: "facebook/nllb-200-3.3B"
parameters: 3.3B
type: "multilingual-translation"
languages: 200+ (includes Pashto)
pashto_codes: ["pbt_Arab", "pbu_Arab"]  # Southern & Northern Pashto

use_cases:
  - Pashto ↔ English translation
  - Pashto ↔ Dari translation
  - Pashto ↔ Urdu translation
  - Pashto ↔ Arabic translation

performance:
  bleu_score: "25+ for Pashto-English"
  quality: "Production ready"
  speed: "Fast inference"

fine_tuning:
  recommended: true
  dataset_size: "50K+ Pashto-English pairs"
  training_time: "6-12 hours on A100"
  memory_required: "16GB VRAM"
```

### 2. M2M-100 (Many-to-Many)
```yaml
model_id: "facebook/m2m100_1.2B"
parameters: 1.2B
type: "many-to-many-translation"
languages: 100 (includes Pashto)
pashto_code: "ps"

use_cases:
  - Direct Pashto to any language
  - No English pivot required
  - Lightweight translation

performance:
  bleu_score: "20+ for Pashto pairs"
  quality: "Good for most use cases"
  speed: "Very fast"

deployment:
  cpu_inference: "8GB RAM"
  gpu_inference: "6GB VRAM"
  mobile: "Suitable for mobile"
```

### 3. mT5-Large (Multilingual T5)
```yaml
model_id: "google/mt5-large"
parameters: 1.2B
type: "text-to-text-transfer"
languages: 101 (includes Pashto)

use_cases:
  - Custom translation fine-tuning
  - Translation with context
  - Domain-specific translation

fine_tuning:
  recommended: true
  technique: "Task-specific fine-tuning"
  dataset_format: "text-to-text"
  specialization: "Technical, literary, news"
```

## Smaller Translation Models

### 4. OPUS-MT Models
```yaml
model_family: "Helsinki-NLP/opus-mt-*"
parameters: 77M - 300M
type: "bilingual-translation"

available_pairs:
  - "Helsinki-NLP/opus-mt-en-ps"  # English to Pashto
  - "Helsinki-NLP/opus-mt-ps-en"  # Pashto to English

use_cases:
  - Fast bilingual translation
  - Edge deployment
  - API rate limiting backup
```

## Custom Translation Models

### ZamAI Pashto-English Translator
```yaml
base_model: "facebook/nllb-200-3.3B"
custom_id: "tasal9/zamai-pashto-english-translator"

fine_tuning_config:
  dataset: "pashto_english_parallel_50k"
  technique: "Full fine-tuning"
  learning_rate: 5e-5
  batch_size: 8
  epochs: 5

specialization:
  - News and current events
  - Religious and cultural texts
  - Technical documentation
  - Literature and poetry
  - Conversational language

quality_improvements:
  - Afghan cultural context
  - Proper noun handling
  - Idiomatic expressions
  - Regional dialect support
```

### ZamAI Pashto-Dari Translator
```yaml
base_model: "facebook/nllb-200-3.3B"
custom_id: "tasal9/zamai-pashto-dari-translator"

specialization:
  - Afghanistan's bilingual context
  - Government documents
  - Educational materials
  - Media content
```

## Translation Pipeline Configurations

### High Quality (Slow)
```python
translation_config = {
    "max_length": 512,
    "num_beams": 5,
    "early_stopping": True,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 3,
    "temperature": 1.0
}
```

### Balanced (Medium Speed)
```python
translation_config = {
    "max_length": 256,
    "num_beams": 3,
    "early_stopping": True,
    "length_penalty": 0.8,
    "do_sample": False
}
```

### Fast (Real-time)
```python
translation_config = {
    "max_length": 128,
    "num_beams": 1,  # Greedy decoding
    "early_stopping": False,
    "do_sample": False
}
```

## Language Codes Reference

### Pashto Language Codes
```yaml
# ISO 639-1/2/3 codes
iso_639_1: "ps"
iso_639_2: "pus"
iso_639_3: "pus"

# NLLB-200 specific codes
nllb_southern: "pbt_Arab"  # Southern Pashto (Kandahar)
nllb_northern: "pbu_Arab"  # Northern Pashto (Peshawar)

# Script variants
script_arabic: "ps_Arab"   # Standard Arabic script
script_latin: "ps_Latn"    # Latin transliteration
```

### Related Languages
```yaml
dari_persian:
  nllb: "prs_Arab"
  iso: "prs"
  
urdu:
  nllb: "urd_Arab"
  iso: "ur"
  
arabic:
  nllb: "arb_Arab"
  iso: "ar"
```

## Quality Assessment

### Translation Quality Metrics
```python
# BLEU Score expectations
quality_benchmarks = {
    "excellent": "> 30 BLEU",
    "good": "20-30 BLEU", 
    "acceptable": "15-20 BLEU",
    "poor": "< 15 BLEU"
}

# Pashto-specific challenges
challenges = [
    "Complex morphology",
    "SOV word order vs English SVO",
    "Arabic script right-to-left",
    "Cultural context dependency",
    "Limited training data"
]
```

### Evaluation Datasets
```yaml
test_sets:
  - name: "Pashto-English News"
    size: "1000 pairs"
    domain: "journalism"
    
  - name: "Pashto-English Conversations"
    size: "500 pairs"
    domain: "informal"
    
  - name: "Pashto-English Literature"
    size: "200 pairs"
    domain: "poetry/prose"
```

## Deployment Strategies

### Multi-Model Ensemble
```python
# Combine multiple models for better quality
ensemble_config = {
    "primary": "tasal9/zamai-pashto-english-translator",
    "fallback": "facebook/nllb-200-3.3B", 
    "fast": "Helsinki-NLP/opus-mt-ps-en",
    
    "routing_logic": "quality_first",
    "confidence_threshold": 0.8
}
```

### A/B Testing Setup
```python
# Test different models in production
ab_test_config = {
    "model_a": "custom_fine_tuned",
    "model_b": "base_nllb_200",
    "traffic_split": 0.5,
    "metrics": ["bleu_score", "user_rating", "response_time"]
}
```
