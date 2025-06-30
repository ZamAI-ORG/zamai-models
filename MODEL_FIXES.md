# ZamAI Model Integration Fixes

Based on debugging results, here are recommended fixes:

## tasal9/pashto-base-bloom
- ❌ Failing methods:
  - direct_api: HTTP 404
  - inference_client: 

## tasal9/pashto-bloom-base
- ❌ Failing methods:
  - direct_api: HTTP 404
  - inference_client: 

## tasal9/ZamAI-LIama3-Pashto
- ⚠️  Model is private - make public or update access tokens
## tasal9/Multilingual-ZamAI-Embeddings
- ❌ Failing methods:
  - direct_api: HTTP 404
  - inference_client: InferenceClient.feature_extraction() got an unexpected keyword argument 'inputs'

## tasal9/ZamAI-Mistral-7B-Pashto
- ❌ Failing methods:
  - direct_api: HTTP 404
  - inference_client: 
