#!/usr/bin/env python3
"""
Check existing spaces and organize them for testing vs training
"""

import os
from huggingface_hub import HfApi, list_models, list_spaces
import json

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def get_existing_models_and_spaces():
    """Get all existing models and spaces"""
    token = read_hf_token()
    api = HfApi(token=token)
    
    print("🔍 Checking existing models and spaces...")
    
    # Get models
    models = list(list_models(author="tasal9", token=token))
    model_names = [model.modelId for model in models]
    
    print(f"\n📦 Found {len(model_names)} models:")
    for model in model_names:
        print(f"   - {model}")
    
    # Get spaces
    spaces = list(list_spaces(author="tasal9", token=token))
    space_info = []
    
    print(f"\n🚀 Found {len(spaces)} spaces:")
    for space in spaces:
        space_info.append({
            'id': space.id,
            'name': space.id.split('/')[-1],
            'full_name': space.id
        })
        print(f"   - {space.id}")
    
    return model_names, space_info

def analyze_space_code(space_id, token):
    """Download and analyze space code for bugs"""
    api = HfApi(token=token)
    
    print(f"\n🔍 Analyzing {space_id}...")
    
    try:
        # Try to get the files
        files = api.list_repo_files(repo_id=space_id, repo_type="space", token=token)
        print(f"   Files: {files}")
        
        analysis = {
            'space_id': space_id,
            'files': files,
            'issues': [],
            'recommendations': []
        }
        
        # Check for essential files
        if 'app.py' not in files:
            analysis['issues'].append("Missing app.py file")
        if 'requirements.txt' not in files:
            analysis['issues'].append("Missing requirements.txt file")
        if 'README.md' not in files:
            analysis['issues'].append("Missing README.md file")
        
        # Download and check app.py if it exists
        if 'app.py' in files:
            try:
                app_content = api.hf_hub_download(
                    repo_id=space_id,
                    filename="app.py",
                    repo_type="space",
                    token=token
                )
                
                with open(app_content, 'r') as f:
                    code = f.read()
                
                # Check for common issues
                if 'import spaces' not in code:
                    analysis['issues'].append("Missing 'import spaces' for ZeroGPU")
                
                if '@spaces.GPU' not in code:
                    analysis['issues'].append("Missing @spaces.GPU decorator")
                
                if 'torch_dtype' not in code and 'transformers' in code:
                    analysis['recommendations'].append("Consider adding torch_dtype=torch.float16 for efficiency")
                
                if 'device_map' not in code and 'transformers' in code:
                    analysis['recommendations'].append("Consider adding device_map='auto'")
                
                if 'try:' not in code:
                    analysis['issues'].append("Missing error handling")
                
            except Exception as e:
                analysis['issues'].append(f"Could not analyze app.py: {str(e)}")
        
        # Check requirements.txt
        if 'requirements.txt' in files:
            try:
                req_content = api.hf_hub_download(
                    repo_id=space_id,
                    filename="requirements.txt",
                    repo_type="space",
                    token=token
                )
                
                with open(req_content, 'r') as f:
                    requirements = f.read()
                
                if 'spaces' not in requirements:
                    analysis['issues'].append("Missing 'spaces' in requirements.txt")
                
                if 'gradio' not in requirements:
                    analysis['issues'].append("Missing 'gradio' in requirements.txt")
                
            except Exception as e:
                analysis['issues'].append(f"Could not analyze requirements.txt: {str(e)}")
        
        return analysis
        
    except Exception as e:
        return {
            'space_id': space_id,
            'error': str(e),
            'issues': [f"Could not access space: {str(e)}"]
        }

def categorize_spaces(models, spaces):
    """Categorize spaces as testing or training based on names"""
    
    categorized = {}
    
    for model in models:
        model_short = model.replace('tasal9/', '').lower()
        categorized[model] = {
            'testing_spaces': [],
            'training_spaces': [],
            'uncategorized_spaces': []
        }
    
    for space in spaces:
        space_name = space['name'].lower()
        matched_model = None
        
        # Try to match space to model
        for model in models:
            model_keywords = model.replace('tasal9/', '').lower().split('-')
            
            # Check if space name contains model keywords
            matches = sum(1 for keyword in model_keywords if keyword in space_name)
            if matches >= 2:  # At least 2 keywords match
                matched_model = model
                break
        
        if matched_model:
            # Determine if it's for testing or training
            if any(word in space_name for word in ['train', 'training', 'finetune', 'fine-tune', 'ft']):
                categorized[matched_model]['training_spaces'].append(space)
            elif any(word in space_name for word in ['test', 'demo', 'chat', 'inference', 'api']):
                categorized[matched_model]['testing_spaces'].append(space)
            else:
                categorized[matched_model]['uncategorized_spaces'].append(space)
    
    return categorized

def main():
    print("🔍 Analyzing ZamAI Spaces Organization")
    print("=" * 60)
    
    token = read_hf_token()
    
    # Get existing models and spaces
    models, spaces = get_existing_models_and_spaces()
    
    # Categorize spaces
    categorized = categorize_spaces(models, spaces)
    
    print("\n📊 SPACE ORGANIZATION ANALYSIS")
    print("=" * 60)
    
    all_analyses = []
    
    for model, space_groups in categorized.items():
        print(f"\n🤖 Model: {model}")
        print(f"   📝 Testing spaces: {len(space_groups['testing_spaces'])}")
        for space in space_groups['testing_spaces']:
            print(f"      - {space['full_name']}")
        
        print(f"   🏋️  Training spaces: {len(space_groups['training_spaces'])}")
        for space in space_groups['training_spaces']:
            print(f"      - {space['full_name']}")
        
        print(f"   ❓ Uncategorized spaces: {len(space_groups['uncategorized_spaces'])}")
        for space in space_groups['uncategorized_spaces']:
            print(f"      - {space['full_name']}")
        
        # Analyze code for each space
        all_spaces = (space_groups['testing_spaces'] + 
                     space_groups['training_spaces'] + 
                     space_groups['uncategorized_spaces'])
        
        for space in all_spaces:
            analysis = analyze_space_code(space['full_name'], token)
            analysis['model'] = model
            analysis['category'] = ('testing' if space in space_groups['testing_spaces'] 
                                  else 'training' if space in space_groups['training_spaces']
                                  else 'uncategorized')
            all_analyses.append(analysis)
    
    # Summary of issues
    print("\n🐛 CODE ANALYSIS SUMMARY")
    print("=" * 60)
    
    total_issues = 0
    spaces_with_issues = 0
    
    for analysis in all_analyses:
        if analysis.get('issues'):
            spaces_with_issues += 1
            total_issues += len(analysis['issues'])
            
            print(f"\n❌ {analysis['space_id']} ({analysis.get('category', 'unknown')}):")
            for issue in analysis['issues']:
                print(f"   • {issue}")
            
            if analysis.get('recommendations'):
                print("   💡 Recommendations:")
                for rec in analysis['recommendations']:
                    print(f"   • {rec}")
    
    # Save detailed analysis
    with open('/workspaces/ZamAI-Pro-Models/space_analysis.json', 'w') as f:
        json.dump({
            'models': models,
            'spaces': spaces,
            'categorized': categorized,
            'analyses': all_analyses,
            'summary': {
                'total_models': len(models),
                'total_spaces': len(spaces),
                'spaces_with_issues': spaces_with_issues,
                'total_issues': total_issues
            }
        }, f, indent=2)
    
    print(f"\n📈 SUMMARY")
    print("=" * 60)
    print(f"📦 Total models: {len(models)}")
    print(f"🚀 Total spaces: {len(spaces)}")
    print(f"❌ Spaces with issues: {spaces_with_issues}")
    print(f"🐛 Total issues found: {total_issues}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 60)
    
    missing_spaces = []
    for model, space_groups in categorized.items():
        if len(space_groups['testing_spaces']) == 0:
            missing_spaces.append(f"Create testing space for {model}")
        if len(space_groups['training_spaces']) == 0:
            missing_spaces.append(f"Create training space for {model}")
    
    if missing_spaces:
        print("🏗️  Missing spaces to create:")
        for missing in missing_spaces:
            print(f"   • {missing}")
    
    if spaces_with_issues > 0:
        print(f"🔧 Fix {total_issues} issues in {spaces_with_issues} spaces")
    
    print(f"\n💾 Detailed analysis saved to: space_analysis.json")

if __name__ == "__main__":
    main()
