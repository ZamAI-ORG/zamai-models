# 🗣️ Language Models for Pashto Chat & Text Generation

## Primary Models

### 1. Llama 3.1 8B Instruct
```yaml
model_id: "meta-llama/Llama-3.1-8B-Instruct"
parameters: 8B
type: "instruction-following"
languages: ["en", "multilingual"]
use_cases:
  - Pashto conversational AI
  - Question answering in Pashto
  - Text completion
  - Creative writing assistance

fine_tuning:
  recommended: true
  dataset_size: "10K+ Pashto conversations"
  training_time: "4-8 hours on A100"
  memory_required: "24GB VRAM"

deployment:
  cpu_inference: "64GB RAM recommended"
  gpu_inference: "16GB VRAM minimum"
  quantization: "4-bit, 8-bit supported"
```

### 2. Mistral 7B Instruct v0.3
```yaml
model_id: "mistralai/Mistral-7B-Instruct-v0.3"
parameters: 7B
type: "instruction-following"
languages: ["en", "fr", "de", "es", "it"]
use_cases:
  - Pashto instruction following
  - Code generation with Pashto comments
  - Technical documentation in Pashto

fine_tuning:
  recommended: true
  dataset_size: "5K+ Pashto instructions"
  training_time: "3-6 hours on A100"
  memory_required: "20GB VRAM"

deployment:
  cpu_inference: "32GB RAM recommended"
  gpu_inference: "12GB VRAM minimum"
  quantization: "Excellent 4-bit performance"
```

### 3. Qwen2.5 7B Instruct
```yaml
model_id: "Qwen/Qwen2.5-7B-Instruct"
parameters: 7B
type: "instruction-following"
languages: ["zh", "en", "multilingual"]
use_cases:
  - Multilingual Pashto support
  - Math and reasoning in Pashto
  - Long context conversations

fine_tuning:
  recommended: true
  dataset_size: "8K+ Pashto examples"
  training_time: "4-7 hours on A100"
  memory_required: "22GB VRAM"

special_features:
  - Extended context length (32K tokens)
  - Strong reasoning capabilities
  - Good multilingual transfer
```

## Lightweight Options

### 4. Phi-3.5 Mini Instruct
```yaml
model_id: "microsoft/Phi-3.5-mini-instruct"
parameters: 3.8B
type: "small-language-model"
languages: ["en", "multilingual"]
use_cases:
  - Edge deployment
  - Mobile Pashto chat
  - Resource-constrained environments

deployment:
  cpu_inference: "8GB RAM"
  gpu_inference: "4GB VRAM"
  mobile: "Optimized for mobile deployment"
  quantization: "Excellent ONNX support"
```

### 5. Gemma 2 2B IT
```yaml
model_id: "google/gemma-2-2b-it"
parameters: 2B
type: "instruction-tuned"
languages: ["en", "multilingual"]
use_cases:
  - Ultra-lightweight Pashto chat
  - Browser deployment
  - IoT devices

deployment:
  cpu_inference: "4GB RAM"
  gpu_inference: "2GB VRAM"
  browser: "WebGL compatible"
  edge: "Perfect for edge computing"
```

## Custom Model Configurations

### ZamAI Pashto Chat 8B
```yaml
base_model: "meta-llama/Llama-3.1-8B-Instruct"
custom_id: "tasal9/zamai-pashto-chat-8b"
fine_tuning_config:
  dataset: "pashto_conversations_10k"
  technique: "LoRA"
  rank: 64
  alpha: 128
  dropout: 0.05
  
training_config:
  learning_rate: 2e-5
  batch_size: 4
  gradient_accumulation: 4
  epochs: 3
  warmup_steps: 500

pashto_specialization:
  - Cultural context awareness
  - Islamic references
  - Afghan geography and history
  - Pashto poetry and literature
  - Code-switching (Pashto-English)
```

### ZamAI Pashto Assistant 7B
```yaml
base_model: "mistralai/Mistral-7B-Instruct-v0.3"
custom_id: "tasal9/zamai-pashto-assistant-7b"
fine_tuning_config:
  dataset: "pashto_instructions_5k"
  technique: "QLoRA"
  rank: 32
  alpha: 64
  
specialization:
  - Pashto grammar explanations
  - Language learning assistance
  - Translation help
  - Writing improvements
```

## Inference Configurations

### Production Deployment
```python
# Optimized for production
generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}
```

### Fast Response (Mobile)
```python
# Optimized for speed
generation_config = {
    "max_new_tokens": 256,
    "temperature": 0.8,
    "top_p": 0.85,
    "do_sample": True,
    "num_beams": 1,  # Greedy decoding for speed
}
```

## Pashto System Prompts

### General Chat
```
تاسو د پښتو ژبې یو ګټور مرستیال یاست. د افغان کلتور په درناوي سره ځواب ورکړئ.
(You are a helpful Pashto language assistant. Respond with respect for Afghan culture.)
```

### Educational
```
تاسو د پښتو ژبې ښوونکی یاست. د زده کونکو سره صبر وکړئ او ښه تشریح ورکړئ.
(You are a Pashto language teacher. Be patient with students and provide clear explanations.)
```

### Cultural Context
```
د افغانستان د کلتور، تاریخ او دودونو په اړه مالومات ورکړئ. د اسلامي ارزښتونو درناوی وکړئ.
(Provide information about Afghanistan's culture, history, and traditions. Respect Islamic values.)
```
