# HF Inference Integration Guide for ZamAI

This guide shows how to integrate Hugging Face Inference into your ZamAI project for testing, development, and production deployment.

## What is HF Inference?

HF Inference provides several ways to run ML models:

1. **Inference API** - Free tier for testing with rate limits
2. **Inference Endpoints** - Dedicated endpoints for production
3. **Serverless Inference** - Pay-per-request model serving

## Quick Start

### 1. Install Dependencies

```bash
pip install huggingface-hub[inference] aiohttp
```

### 2. Set Up Authentication

Make sure your HF token is in `HF-Token.txt` or set as environment variable:

```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

### 3. Test Basic Inference

```python
from scripts.inference.hf_inference_client import ZamAIInferenceClient

client = ZamAIInferenceClient()

# Test chat completion
response = client.chat_completion(
    model_id="microsoft/DialoGPT-medium",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Integration with Your Models

### Pashto Chat Model

Once you deploy your `zamai-pashto-chat-8b` model:

```python
# Test your fine-tuned model
client = ZamAIInferenceClient()
response = client.test_pashto_chat("سلام ورور، ستاسو څنګه یاست؟")
print(response)
```

### Model Configuration Integration

The inference client automatically reads your model configs:

```python
# configs/pashto_chat_config.json
{
  "hub_model_id": "tasal9/zamai-pashto-chat-8b",
  "system_prompts": {
    "general": "تاسو د پښتو ژبې یو ګټور مرستیال یاست..."
  }
}
```

## Usage Scenarios

### 1. Development & Testing

```bash
# Test inference API
python scripts/inference/test_hf_inference.py
```

**Benefits:**
- No local GPU needed
- Quick model testing
- Free tier available

### 2. Production Deployment

```bash
# Deploy to Inference Endpoints
python scripts/inference/deploy_endpoints.py
```

**Benefits:**
- Auto-scaling (0 to N replicas)
- Production-grade infrastructure
- Custom domains and SSL

### 3. Integration with Existing Scripts

Update your existing model testing scripts:

```python
# Before (local inference)
from transformers import pipeline
pipe = pipeline("text-generation", model="local_model")

# After (HF Inference)
from scripts.inference.hf_inference_client import ZamAIInferenceClient
client = ZamAIInferenceClient()
response = client.text_generation("tasal9/zamai-pashto-chat-8b", prompt)
```

## Cost Optimization

### Free Tier (Inference API)
- Rate limited
- Good for development
- No cost

### Inference Endpoints
- **Small**: ~$0.30/hour (CPU)
- **Medium**: ~$0.60/hour (Tesla T4)
- **Large**: ~$1.30/hour (A10G)
- **XLarge**: ~$4.50/hour (A100)

### Best Practices
1. Use auto-scaling (min_replica=0)
2. Start with smaller instances
3. Monitor usage with HF dashboard
4. Use serverless for low-traffic models

## Supported Use Cases for ZamAI

### 1. Text Generation
```python
# Your Pashto language models
client.text_generation(
    model_id="tasal9/zamai-pashto-chat-8b",
    prompt="د افغانستان د تاریخ په اړه"
)
```

### 2. Chat Completion
```python
# Conversational AI
messages = [
    {"role": "system", "content": "Pashto assistant system prompt"},
    {"role": "user", "content": "User question in Pashto"}
]
client.chat_completion(model_id, messages)
```

### 3. Embeddings
```python
# Semantic search and similarity
embeddings = client.embeddings(
    model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    texts=["پښتو متن", "English text"]
)
```

### 4. Speech Recognition
```python
# Audio transcription
transcript = client.automatic_speech_recognition(
    model_id="openai/whisper-large-v3",
    audio_path="pashto_audio.wav"
)
```

## Deployment Workflow

### 1. Train Your Model
```bash
python fine-tuning/train_pashto_chat.py
```

### 2. Push to Hub
```python
# Already configured in your training script
push_to_hub=True
hub_model_id="tasal9/zamai-pashto-chat-8b"
```

### 3. Deploy Endpoint
```bash
python scripts/inference/deploy_endpoints.py
```

### 4. Test Production Endpoint
```python
endpoint_url = "https://your-endpoint.endpoints.huggingface.co"
response = client.test_endpoint(endpoint_url, "سلام ورور")
```

## Monitoring & Management

### View Endpoints
```python
manager = ZamAIEndpointManager()
endpoints = manager.list_endpoints()
```

### Check Status
```python
status = manager.get_endpoint_status("zamai-pashto-chat")
```

### Auto-scaling Configuration
```json
{
  "scaling": {
    "minReplica": 0,  // Scale to zero when idle
    "maxReplica": 3   // Handle traffic spikes
  }
}
```

## Integration with Your Existing Code

### Update test_models.py
```python
# Add HF Inference testing
from scripts.inference.hf_inference_client import ZamAIInferenceClient

def test_hf_inference():
    client = ZamAIInferenceClient()
    # Test your deployed models
    # Add results to existing test reports
```

### Update model_manager.py
```python
# Add endpoint management
from scripts.inference.deploy_endpoints import ZamAIEndpointManager

class ModelManager:
    def __init__(self):
        self.endpoint_manager = ZamAIEndpointManager()
    
    def deploy_model(self, model_config):
        # Deploy to HF Endpoints
        pass
```

## Next Steps

1. **Test the scripts**: Run the inference client with existing models
2. **Deploy your Pashto model**: Use the deployment script after training
3. **Monitor costs**: Start with free tier, then upgrade as needed
4. **Integrate**: Update your existing scripts to use HF Inference
5. **Scale**: Use auto-scaling endpoints for production

## Troubleshooting

### Common Issues
1. **Token errors**: Ensure HF token has write permissions
2. **Model not found**: Verify model is public or you have access
3. **Rate limits**: Use endpoints for high-volume usage
4. **Cold starts**: Endpoints may take 30-60s to wake up

### Support Resources
- [HF Inference Documentation](https://huggingface.co/docs/inference-endpoints/)
- [Pricing Calculator](https://huggingface.co/pricing)
- Your model configs in `configs/` directory
