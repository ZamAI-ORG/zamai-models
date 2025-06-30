# 🎤 Speech Models for Pashto

## Speech Recognition (ASR) Models

### 1. Whisper Large v3 (Primary)
```yaml
model_id: "openai/whisper-large-v3"
parameters: 1.55B
type: "automatic-speech-recognition"
languages: 99+ (includes Pashto)
pashto_support: "Native support"

capabilities:
  - Pashto speech recognition
  - Noise robustness
  - Accent adaptation
  - Long-form transcription
  - Real-time streaming

performance:
  word_error_rate: "15-25% (Pashto)"
  processing_speed: "Real-time on GPU"
  audio_quality: "Handles 8kHz - 48kHz"

fine_tuning:
  recommended: true
  dataset_size: "100+ hours Pashto audio"
  training_time: "12-24 hours on A100"
  memory_required: "24GB VRAM"

deployment:
  cpu_inference: "16GB RAM"
  gpu_inference: "8GB VRAM"
  streaming: "Supported with modifications"
```

### 2. Wav2Vec2 XLS-R 300M
```yaml
model_id: "facebook/wav2vec2-xls-r-300m"
parameters: 300M
type: "self-supervised-speech"
languages: 128 (includes Pashto)

use_cases:
  - Pashto ASR fine-tuning
  - Low-resource adaptation
  - Feature extraction

fine_tuning:
  recommended: true
  technique: "CTC fine-tuning"
  dataset_size: "10+ hours Pashto audio"
  training_time: "4-8 hours on V100"

advantages:
  - Smaller model size
  - Good for fine-tuning
  - Open source
```

### 3. SpeechT5 ASR
```yaml
model_id: "microsoft/speecht5_asr"
parameters: 144M
type: "unified-speech-text"
languages: "Multilingual"

use_cases:
  - Lightweight Pashto ASR
  - Mobile deployment
  - Custom adaptation

deployment:
  cpu_inference: "4GB RAM"
  gpu_inference: "2GB VRAM"
  mobile: "Optimized for edge"
```

## Text-to-Speech (TTS) Models

### 1. SpeechT5 TTS
```yaml
model_id: "microsoft/speecht5_tts"
parameters: 144M
type: "text-to-speech"
languages: "English (can be adapted)"

adaptation_for_pashto:
  - Voice cloning with Pashto speakers
  - Phoneme mapping for Pashto sounds
  - Prosody adaptation

fine_tuning:
  dataset_required: "10+ hours Pashto speech"
  speaker_adaptation: "1-5 minutes per speaker"
  quality: "Near native with good data"
```

### 2. Facebook MMS TTS
```yaml
model_id: "facebook/mms-tts"
parameters: "Varies by language"
type: "massively-multilingual-speech"
languages: 1100+ (includes Pashto)

pashto_variants:
  - Northern Pashto (Peshawar)
  - Southern Pashto (Kandahar)

quality:
  intelligibility: "Good"
  naturalness: "Moderate"
  accent: "Generic Pashto"
```

### 3. VITS (Conditional Variational Autoencoder)
```yaml
model_id: "facebook/mms-tts-pbt"  # Pashto specific
parameters: 113M
type: "end-to-end-tts"
language: "Pashto (Southern)"

capabilities:
  - Natural prosody
  - Emotional expression
  - Fast inference

deployment:
  gpu_inference: "2GB VRAM"
  cpu_inference: "Possible but slow"
  real_time_factor: "0.1x (very fast)"
```

## Custom Speech Models

### ZamAI Pashto ASR
```yaml
base_model: "openai/whisper-large-v3"
custom_id: "tasal9/zamai-pashto-asr"

fine_tuning_config:
  dataset: "pashto_audio_dataset_100h"
  technique: "LoRA adaptation"
  target_modules: ["encoder", "decoder"]
  rank: 32

specialization:
  - Afghan accent adaptation
  - Noisy environment robustness
  - Domain-specific vocabulary
  - Code-switching (Pashto-English)
  - Religious and cultural terms

training_data:
  - BBC Pashto broadcasts
  - VOA Pashto programs
  - Pashto conversations
  - Educational content
  - Cultural programs
```

### ZamAI Pashto TTS
```yaml
base_model: "microsoft/speecht5_tts"
custom_id: "tasal9/zamai-pashto-tts"

voice_characteristics:
  - Male Afghan speaker
  - Female Afghan speaker
  - Regional accent variants
  - Emotional expressions

fine_tuning_approach:
  - Voice cloning from Pashto speakers
  - Phoneme-level adaptation
  - Prosody modeling for Pashto
  - Cultural pronunciation norms
```

## Audio Processing Pipeline

### Preprocessing for Pashto ASR
```python
audio_preprocessing = {
    "sample_rate": 16000,
    "channels": 1,  # Mono
    "bit_depth": 16,
    "format": "wav",
    
    "noise_reduction": True,
    "normalization": "rms",
    "silence_removal": True,
    
    # Pashto-specific
    "energy_threshold": 300,
    "pause_threshold": 0.8,  # Longer pauses in Pashto
}
```

### Postprocessing for Pashto Text
```python
text_postprocessing = {
    "punctuation_restoration": True,
    "capitalization": True,
    "number_normalization": True,  # Pashto numbers
    "diacritic_restoration": True,  # Arabic diacritics
    
    # Pashto-specific
    "script_normalization": "arabic",
    "transliteration": "optional",
    "code_switch_detection": True,
}
```

## Streaming Speech Recognition

### Real-time Pashto ASR
```python
streaming_config = {
    "chunk_length_s": 30,
    "stride_length_s": 5,
    "language": "ps",  # Pashto
    "task": "transcribe",
    
    # Real-time optimizations
    "return_timestamps": True,
    "word_timestamps": True,
    "vad_filter": True,  # Voice Activity Detection
}
```

### WebRTC Integration
```javascript
// Browser-based Pashto speech recognition
const streamingConfig = {
    sampleRate: 16000,
    channels: 1,
    language: 'ps',
    continuous: true,
    interimResults: true,
    maxAlternatives: 1
};
```

## Voice Activity Detection (VAD)

### Silero VAD for Pashto
```yaml
model_id: "silero/silero-vad"
use_case: "Detect Pashto speech in audio"

configuration:
  threshold: 0.5
  min_speech_duration_ms: 250
  max_speech_duration_s: 30
  min_silence_duration_ms: 100
  window_size_samples: 1536  # 96ms at 16kHz
```

## Performance Benchmarks

### ASR Quality Metrics
```yaml
whisper_large_v3_pashto:
  word_error_rate: "18.5%"
  character_error_rate: "8.2%"
  real_time_factor: "0.3x"
  
wav2vec2_xls_r_pashto:
  word_error_rate: "22.1%"
  character_error_rate: "11.5%"
  real_time_factor: "0.1x"
```

### TTS Quality Metrics
```yaml
speecht5_pashto:
  naturalness_mos: "3.8/5.0"
  intelligibility: "95%"
  speaker_similarity: "Good"
  
mms_tts_pashto:
  naturalness_mos: "3.2/5.0"
  intelligibility: "90%"
  speaker_similarity: "Fair"
```

## Deployment Configurations

### Production ASR Setup
```python
production_config = {
    "model": "tasal9/zamai-pashto-asr",
    "device": "cuda",
    "torch_dtype": "float16",
    "use_flash_attention": True,
    
    "generation_kwargs": {
        "language": "ps",
        "task": "transcribe",
        "temperature": 0.0,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6
    }
}
```

### Mobile ASR Setup
```python
mobile_config = {
    "model": "wav2vec2-xls-r-300m-pashto",
    "quantization": "int8",
    "optimize_for_mobile": True,
    "chunk_size": 10,  # seconds
    "overlap": 1,      # second
}
```

## Audio Datasets for Training

### Recommended Pashto Audio Sources
```yaml
public_datasets:
  - name: "Common Voice Pashto"
    hours: "50+"
    quality: "Crowd-sourced"
    url: "commonvoice.mozilla.org"
    
  - name: "OpenSLR Pashto"
    hours: "20+"
    quality: "Professional"
    url: "openslr.org"

broadcast_sources:
  - BBC Pashto Service
  - VOA Pashto
  - Radio Free Afghanistan
  - Afghan TV channels

custom_collection:
  - Pashto conversations
  - Educational content
  - Religious speeches
  - Poetry recitations
  - News broadcasts
```
