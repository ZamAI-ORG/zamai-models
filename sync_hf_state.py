#!/usr/bin/env python3
"""
Sync HuggingFace Models and Spaces with Local Repository
Fetches latest state from HF and updates local documentation
"""

import json
from datetime import datetime
from huggingface_hub import HfApi
from pathlib import Path

def read_hf_token():
    token_file = Path('/workspaces/ZamAI-Pro-Models/HF-Token.txt')
    if token_file.exists():
        return token_file.read_text().strip()
    return None

def fetch_models(api):
    """Fetch all models from HuggingFace"""
    print("🔍 Fetching models from HuggingFace...")
    models = list(api.list_models(author='tasal9'))
    
    model_data = []
    for model in models:
        try:
            info = api.model_info(model.modelId)
            model_data.append({
                'id': model.modelId,
                'created_at': str(info.created_at) if hasattr(info, 'created_at') else 'Unknown',
                'last_modified': str(info.lastModified) if hasattr(info, 'lastModified') else 'Unknown',
                'downloads': info.downloads if hasattr(info, 'downloads') else 0,
                'likes': info.likes if hasattr(info, 'likes') else 0,
                'tags': info.tags if hasattr(info, 'tags') else [],
                'pipeline_tag': info.pipeline_tag if hasattr(info, 'pipeline_tag') else 'unknown',
                'library_name': info.library_name if hasattr(info, 'library_name') else 'unknown'
            })
        except Exception as e:
            print(f"   ⚠️  Could not fetch info for {model.modelId}: {e}")
            model_data.append({
                'id': model.modelId,
                'error': str(e)
            })
    
    print(f"   ✅ Found {len(model_data)} models")
    return model_data

def fetch_spaces(api):
    """Fetch all spaces from HuggingFace"""
    print("\n🔍 Fetching spaces from HuggingFace...")
    spaces = list(api.list_spaces(author='tasal9'))
    
    space_data = []
    for space in spaces:
        try:
            info = api.space_info(space.id)
            space_data.append({
                'id': space.id,
                'created_at': str(info.created_at) if hasattr(info, 'created_at') else 'Unknown',
                'last_modified': str(info.lastModified) if hasattr(info, 'lastModified') else 'Unknown',
                'likes': info.likes if hasattr(info, 'likes') else 0,
                'sdk': info.sdk if hasattr(info, 'sdk') else 'unknown',
                'runtime': info.runtime.stage if hasattr(info, 'runtime') and hasattr(info.runtime, 'stage') else 'unknown',
                'hardware': info.runtime.hardware if hasattr(info, 'runtime') and hasattr(info.runtime, 'hardware') else 'unknown'
            })
        except Exception as e:
            print(f"   ⚠️  Could not fetch info for {space.id}: {e}")
            space_data.append({
                'id': space.id,
                'error': str(e)
            })
    
    print(f"   ✅ Found {len(space_data)} spaces")
    return space_data

def generate_markdown_report(models, spaces):
    """Generate markdown report of current state"""
    report = f"""# 🤗 ZamAI HuggingFace Status Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary

- **Total Models**: {len(models)}
- **Total Spaces**: {len(spaces)}
- **Total Downloads**: {sum(m.get('downloads', 0) for m in models)}
- **Total Likes**: {sum(m.get('likes', 0) for m in models) + sum(s.get('likes', 0) for s in spaces)}

---

## 🤖 Models ({len(models)})

"""
    
    # Sort models by downloads
    sorted_models = sorted(models, key=lambda x: x.get('downloads', 0), reverse=True)
    
    for model in sorted_models:
        if 'error' in model:
            report += f"### ❌ {model['id']}\n"
            report += f"- **Error**: {model['error']}\n\n"
        else:
            report += f"### {model['id']}\n"
            report += f"- **Pipeline**: {model.get('pipeline_tag', 'unknown')}\n"
            report += f"- **Library**: {model.get('library_name', 'unknown')}\n"
            report += f"- **Downloads**: {model.get('downloads', 0)} 📥\n"
            report += f"- **Likes**: {model.get('likes', 0)} ❤️\n"
            report += f"- **Created**: {model.get('created_at', 'Unknown')}\n"
            report += f"- **Last Modified**: {model.get('last_modified', 'Unknown')}\n"
            
            if model.get('tags'):
                report += f"- **Tags**: {', '.join(model['tags'][:5])}\n"
            
            report += f"- **URL**: https://huggingface.co/{model['id']}\n\n"
    
    report += "---\n\n"
    report += f"## 🚀 Spaces ({len(spaces)})\n\n"
    
    for space in spaces:
        if 'error' in space:
            report += f"### ❌ {space['id']}\n"
            report += f"- **Error**: {space['error']}\n\n"
        else:
            report += f"### {space['id']}\n"
            report += f"- **SDK**: {space.get('sdk', 'unknown')}\n"
            report += f"- **Hardware**: {space.get('hardware', 'unknown')}\n"
            report += f"- **Runtime**: {space.get('runtime', 'unknown')}\n"
            report += f"- **Likes**: {space.get('likes', 0)} ❤️\n"
            report += f"- **Created**: {space.get('created_at', 'Unknown')}\n"
            report += f"- **Last Modified**: {space.get('last_modified', 'Unknown')}\n"
            report += f"- **URL**: https://huggingface.co/spaces/{space['id']}\n\n"
    
    report += """---

## 🎯 Next Actions

### For Models:
1. **Promote Popular Models**: Share models with highest downloads
2. **Update Model Cards**: Ensure all models have comprehensive documentation
3. **Add Examples**: Include usage examples in each model card
4. **Tag Optimization**: Add relevant tags for better discoverability

### For Spaces:
1. **ZeroGPU Optimization**: Ensure all spaces use ZeroGPU decorators
2. **Add Examples**: Include sample inputs/outputs
3. **Update Hardware**: Migrate to latest hardware options
4. **Performance Monitoring**: Add analytics and error tracking

### Community Growth:
1. **Social Media**: Share on Twitter, LinkedIn, Reddit
2. **Blog Posts**: Write technical articles about Pashto AI
3. **Demos**: Create video demonstrations
4. **Collaborations**: Partner with Afghan tech communities

---

**🇦🇫 Building AI for Afghanistan and Pashto Language**
"""
    
    return report

def save_json_data(models, spaces):
    """Save raw JSON data"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'models': models,
        'spaces': spaces,
        'summary': {
            'total_models': len(models),
            'total_spaces': len(spaces),
            'total_downloads': sum(m.get('downloads', 0) for m in models),
            'total_likes': sum(m.get('likes', 0) for m in models) + sum(s.get('likes', 0) for s in spaces)
        }
    }
    
    json_file = Path('/workspaces/ZamAI-Pro-Models/hf_current_state.json')
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Saved JSON data to: {json_file}")

def main():
    print("🔄 Syncing HuggingFace State with Local Repository")
    print("=" * 70)
    
    # Initialize HF API
    token = read_hf_token()
    api = HfApi(token=token) if token else HfApi()
    
    # Fetch data
    models = fetch_models(api)
    spaces = fetch_spaces(api)
    
    # Generate report
    print("\n📝 Generating markdown report...")
    report = generate_markdown_report(models, spaces)
    
    # Save markdown report
    report_file = Path('/workspaces/ZamAI-Pro-Models/HF_CURRENT_STATE.md')
    report_file.write_text(report)
    print(f"   ✅ Saved report to: {report_file}")
    
    # Save JSON data
    save_json_data(models, spaces)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"✅ Models: {len(models)}")
    print(f"✅ Spaces: {len(spaces)}")
    print(f"📥 Total Downloads: {sum(m.get('downloads', 0) for m in models)}")
    print(f"❤️  Total Likes: {sum(m.get('likes', 0) for m in models) + sum(s.get('likes', 0) for s in spaces)}")
    print("\n🎯 Next: Review HF_CURRENT_STATE.md for detailed information")
    print("=" * 70)

if __name__ == "__main__":
    main()
