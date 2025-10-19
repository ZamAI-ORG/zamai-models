#!/usr/bin/env python3
"""
Comprehensive Model Testing Suite
Tests all ZamAI models on HuggingFace
"""

import sys
from pathlib import Path
from huggingface_hub import HfApi, InferenceClient
import json
from datetime import datetime

def read_hf_token():
    token_file = Path('/workspaces/ZamAI-Pro-Models/HF-Token.txt')
    return token_file.read_text().strip()

def test_text_generation_model(model_id, client, token):
    """Test a text generation model"""
    print(f"\n   🧪 Testing text generation...")
    
    test_prompts = [
        "سلام، تاسو څنګه یاست؟",  # Pashto: Hello, how are you?
        "د افغانستان پایتخت څه دی؟",  # Pashto: What is the capital of Afghanistan?
        "Hello, can you help me?",
    ]
    
    results = []
    for prompt in test_prompts:
        try:
            response = client.text_generation(
                prompt,
                model=model_id,
                max_new_tokens=50,
                temperature=0.7,
            )
            results.append({
                'prompt': prompt,
                'response': response,
                'status': 'success'
            })
            print(f"      ✅ Prompt: {prompt[:30]}...")
        except Exception as e:
            results.append({
                'prompt': prompt,
                'error': str(e),
                'status': 'failed'
            })
            print(f"      ❌ Prompt failed: {str(e)[:50]}...")
    
    return results

def test_embedding_model(model_id, client, token):
    """Test an embedding model"""
    print(f"\n   🧪 Testing embeddings...")
    
    test_texts = [
        "د افغانستان ښکلی ملک دی",  # Pashto: Afghanistan is a beautiful country
        "Hello, this is a test",
        "Machine learning is amazing"
    ]
    
    results = []
    for text in test_texts:
        try:
            response = client.feature_extraction(text, model=model_id)
            results.append({
                'text': text,
                'embedding_dim': len(response) if isinstance(response, list) else 'unknown',
                'status': 'success'
            })
            print(f"      ✅ Text: {text[:30]}...")
        except Exception as e:
            results.append({
                'text': text,
                'error': str(e),
                'status': 'failed'
            })
            print(f"      ❌ Text failed: {str(e)[:50]}...")
    
    return results

def test_speech_model(model_id, client, token):
    """Test a speech recognition model"""
    print(f"\n   🧪 Testing speech recognition...")
    print(f"      ⚠️  Skipping audio test (requires audio file)")
    
    return [{
        'status': 'skipped',
        'reason': 'Audio testing requires sample files'
    }]

def test_model(model_id, model_info, token):
    """Test a single model"""
    print(f"\n🔬 Testing: {model_id}")
    print(f"   Pipeline: {model_info.get('pipeline_tag', 'unknown')}")
    
    client = InferenceClient(token=token)
    
    results = {
        'model_id': model_id,
        'pipeline_tag': model_info.get('pipeline_tag', 'unknown'),
        'test_timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    try:
        pipeline_tag = model_info.get('pipeline_tag', 'unknown')
        
        if pipeline_tag in ['text-generation', 'text2text-generation']:
            results['tests'] = test_text_generation_model(model_id, client, token)
        elif pipeline_tag in ['feature-extraction', 'sentence-similarity']:
            results['tests'] = test_embedding_model(model_id, client, token)
        elif pipeline_tag in ['automatic-speech-recognition']:
            results['tests'] = test_speech_model(model_id, client, token)
        else:
            print(f"   ⚠️  Unknown pipeline type: {pipeline_tag}")
            results['tests'] = [{
                'status': 'skipped',
                'reason': f'Unsupported pipeline: {pipeline_tag}'
            }]
        
        # Calculate success rate
        successful_tests = sum(1 for t in results['tests'] if t.get('status') == 'success')
        total_tests = len(results['tests'])
        results['success_rate'] = f"{successful_tests}/{total_tests}"
        results['overall_status'] = 'passed' if successful_tests > 0 else 'failed'
        
        print(f"   📊 Success rate: {successful_tests}/{total_tests}")
        
    except Exception as e:
        print(f"   ❌ Testing failed: {e}")
        results['tests'] = [{
            'status': 'error',
            'error': str(e)
        }]
        results['overall_status'] = 'error'
    
    return results

def generate_test_report(all_results):
    """Generate markdown test report"""
    report = f"""# 🧪 ZamAI Models Test Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary

- **Total Models Tested**: {len(all_results)}
- **Passed**: {sum(1 for r in all_results if r.get('overall_status') == 'passed')}
- **Failed**: {sum(1 for r in all_results if r.get('overall_status') == 'failed')}
- **Errors**: {sum(1 for r in all_results if r.get('overall_status') == 'error')}
- **Skipped**: {sum(1 for r in all_results if r.get('overall_status') == 'skipped')}

---

## 🔬 Detailed Results

"""
    
    for result in all_results:
        status_emoji = {
            'passed': '✅',
            'failed': '❌',
            'error': '⚠️',
            'skipped': '⏭️'
        }.get(result.get('overall_status', 'unknown'), '❓')
        
        report += f"### {status_emoji} {result['model_id']}\n\n"
        report += f"- **Pipeline**: {result.get('pipeline_tag', 'unknown')}\n"
        report += f"- **Status**: {result.get('overall_status', 'unknown').upper()}\n"
        report += f"- **Success Rate**: {result.get('success_rate', 'N/A')}\n"
        report += f"- **Tested**: {result.get('test_timestamp', 'Unknown')}\n\n"
        
        if result.get('tests'):
            report += "**Test Details:**\n\n"
            for i, test in enumerate(result['tests'], 1):
                if test.get('status') == 'success':
                    report += f"{i}. ✅ Success"
                    if 'prompt' in test:
                        report += f" - Prompt: `{test['prompt'][:50]}...`"
                    elif 'text' in test:
                        report += f" - Text: `{test['text'][:50]}...`"
                    report += "\n"
                elif test.get('status') == 'failed':
                    report += f"{i}. ❌ Failed - {test.get('error', 'Unknown error')}\n"
                elif test.get('status') == 'skipped':
                    report += f"{i}. ⏭️  Skipped - {test.get('reason', 'No reason provided')}\n"
        
        report += "\n---\n\n"
    
    report += """## 🎯 Recommendations

### For Failed Models:
1. Check model accessibility and permissions
2. Verify model is properly deployed
3. Review model card for usage instructions
4. Test with different inputs

### For Passed Models:
1. Add more comprehensive test cases
2. Test edge cases and error handling
3. Monitor performance metrics
4. Share successful examples in model cards

### General Improvements:
1. Set up automated testing pipeline
2. Add integration tests
3. Create benchmark datasets
4. Monitor model drift

---

**🇦🇫 ZamAI - Tested and Verified**
"""
    
    return report

def main():
    print("🧪 ZamAI Models - Comprehensive Testing Suite")
    print("=" * 70)
    
    token = read_hf_token()
    api = HfApi(token=token)
    
    # Get all models
    print("\n🔍 Fetching models from HuggingFace...")
    models = list(api.list_models(author='tasal9'))
    print(f"   Found {len(models)} models")
    
    # Get model info
    model_info_map = {}
    for model in models:
        try:
            info = api.model_info(model.modelId)
            model_info_map[model.modelId] = {
                'pipeline_tag': info.pipeline_tag if hasattr(info, 'pipeline_tag') else 'unknown',
                'tags': info.tags if hasattr(info, 'tags') else []
            }
        except Exception as e:
            print(f"   ⚠️  Could not fetch info for {model.modelId}: {e}")
            model_info_map[model.modelId] = {'pipeline_tag': 'unknown'}
    
    # Test each model
    all_results = []
    for model in models:
        try:
            result = test_model(model.modelId, model_info_map[model.modelId], token)
            all_results.append(result)
        except Exception as e:
            print(f"   ❌ Unexpected error testing {model.modelId}: {e}")
            all_results.append({
                'model_id': model.modelId,
                'overall_status': 'error',
                'error': str(e)
            })
    
    # Save results
    print("\n💾 Saving results...")
    
    # JSON results
    json_file = Path('/workspaces/ZamAI-Pro-Models/model_test_results.json')
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2)
    print(f"   ✅ Saved JSON to: {json_file}")
    
    # Markdown report
    report = generate_test_report(all_results)
    report_file = Path('/workspaces/ZamAI-Pro-Models/MODEL_TEST_REPORT.md')
    report_file.write_text(report)
    print(f"   ✅ Saved report to: {report_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TESTING SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in all_results if r.get('overall_status') == 'passed')
    failed = sum(1 for r in all_results if r.get('overall_status') == 'failed')
    errors = sum(1 for r in all_results if r.get('overall_status') == 'error')
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Errors: {errors}")
    print(f"📝 Total: {len(all_results)}")
    print("=" * 70)
    
    return 0 if failed == 0 and errors == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
