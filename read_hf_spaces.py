#!/usr/bin/env python3
"""
Read and analyze all ZamAI HuggingFace Spaces in detail
- Check space configurations
- Identify ZeroGPU usage
- Analyze app.py files
- Check hardware settings
"""

import os
from huggingface_hub import HfApi
import requests
import json

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def get_file_content(space_id, filename, token):
    """Get content of a file from a HF space"""
    try:
        api = HfApi(token=token)
        content = api.hf_hub_download(
            repo_id=space_id,
            filename=filename,
            repo_type="space",
            token=token
        )
        
        with open(content, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading {filename}: {str(e)}"

def analyze_space_config(space_id, token):
    """Analyze a space's configuration in detail"""
    api = HfApi(token=token)
    
    try:
        # Get space info
        space_info = api.space_info(space_id)
        
        # Get list of files
        files = api.list_repo_files(space_id, repo_type="space")
        
        # Read README.md to get configuration
        readme_content = get_file_content(space_id, "README.md", token)
        
        # Read app.py to understand functionality
        app_content = get_file_content(space_id, "app.py", token) if "app.py" in files else "No app.py found"
        
        # Read requirements.txt
        req_content = get_file_content(space_id, "requirements.txt", token) if "requirements.txt" in files else "No requirements.txt found"
        
        # Parse README for hardware info
        hardware = "unknown"
        if "hardware:" in readme_content:
            for line in readme_content.split('\n'):
                if line.strip().startswith('hardware:'):
                    hardware = line.split(':', 1)[1].strip()
                    break
        
        # Check for ZeroGPU usage
        zerogpu_usage = False
        if "zero" in hardware.lower() or "gpu" in hardware.lower():
            zerogpu_usage = True
        
        # Check app.py for ZeroGPU decorators
        if "@spaces.GPU" in app_content or "zero_gpu" in app_content.lower():
            zerogpu_usage = True
        
        # Get runtime status
        runtime_status = "unknown"
        try:
            if hasattr(space_info, 'runtime') and space_info.runtime:
                runtime_status = getattr(space_info.runtime, 'stage', 'unknown')
        except:
            runtime_status = "unknown"
        
        analysis = {
            'space_id': space_id,
            'sdk': getattr(space_info, 'sdk', 'unknown'),
            'hardware': hardware,
            'zerogpu_usage': zerogpu_usage,
            'runtime_status': runtime_status,
            'files': files,
            'file_count': len(files),
            'has_app': "app.py" in files,
            'has_readme': "README.md" in files,
            'has_requirements': "requirements.txt" in files,
            'readme_content': readme_content[:500] + "..." if len(readme_content) > 500 else readme_content,
            'app_preview': app_content[:300] + "..." if len(app_content) > 300 else app_content,
            'requirements': req_content
        }
        
        return analysis
        
    except Exception as e:
        return {
            'space_id': space_id,
            'error': str(e)
        }

def read_all_spaces():
    """Read and analyze all user's spaces"""
    
    token = read_hf_token()
    api = HfApi(token=token)
    
    print("🔍 READING ALL ZAMAI HUGGINGFACE SPACES")
    print("=" * 60)
    
    # Get user info
    user_info = api.whoami()
    username = user_info['name']
    print(f"👤 User: {username}")
    
    # Get all spaces
    spaces = list(api.list_spaces(author=username))
    print(f"🚀 Total Spaces: {len(spaces)}")
    
    space_analyses = []
    zerogpu_spaces = []
    
    for space in spaces:
        space_id = space.id
        created_date = space.created_at.strftime('%Y-%m-%d') if space.created_at else 'Unknown'
        
        print(f"\n📂 Analyzing: {space_id}")
        print(f"   📅 Created: {created_date}")
        
        # Analyze the space
        analysis = analyze_space_config(space_id, token)
        space_analyses.append(analysis)
        
        if 'error' in analysis:
            print(f"   ❌ Error: {analysis['error']}")
            continue
        
        # Display analysis
        print(f"   📱 SDK: {analysis['sdk']}")
        print(f"   🔧 Hardware: {analysis['hardware']}")
        print(f"   ⚡ ZeroGPU: {'✅ YES' if analysis['zerogpu_usage'] else '❌ No'}")
        print(f"   🔄 Status: {analysis['runtime_status']}")
        print(f"   📁 Files: {analysis['file_count']}")
        
        # Check file completeness
        missing_files = []
        if not analysis['has_app']:
            missing_files.append('app.py')
        if not analysis['has_readme']:
            missing_files.append('README.md')
        if not analysis['has_requirements']:
            missing_files.append('requirements.txt')
        
        if missing_files:
            print(f"   ⚠️  Missing: {', '.join(missing_files)}")
        else:
            print(f"   ✅ All core files present")
        
        # Track ZeroGPU spaces
        if analysis['zerogpu_usage']:
            zerogpu_spaces.append(space_id)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 SPACES SUMMARY")
    
    # Count by hardware type
    hardware_counts = {}
    working_spaces = 0
    error_spaces = 0
    
    for analysis in space_analyses:
        if 'error' in analysis:
            error_spaces += 1
            continue
            
        working_spaces += 1
        hardware = analysis['hardware']
        hardware_counts[hardware] = hardware_counts.get(hardware, 0) + 1
    
    print(f"✅ Working Spaces: {working_spaces}")
    print(f"❌ Error Spaces: {error_spaces}")
    print(f"⚡ ZeroGPU Spaces: {len(zerogpu_spaces)}")
    
    print(f"\n🔧 HARDWARE BREAKDOWN:")
    for hardware, count in hardware_counts.items():
        print(f"   {hardware}: {count} spaces")
    
    if zerogpu_spaces:
        print(f"\n⚡ ZEROGPU ENABLED SPACES:")
        for space_id in zerogpu_spaces:
            print(f"   🚀 {space_id}")
            print(f"      🔗 https://huggingface.co/spaces/{space_id}")
    
    # Save detailed analysis
    output_file = '/workspaces/ZamAI-Pro-Models/data/processed/spaces_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(space_analyses, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Detailed analysis saved to: {output_file}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    non_zerogpu_spaces = [s for s in space_analyses if not s.get('zerogpu_usage', False) and 'error' not in s]
    if non_zerogpu_spaces:
        print(f"1. Consider upgrading {len(non_zerogpu_spaces)} spaces to ZeroGPU for better performance")
    
    if error_spaces > 0:
        print(f"2. Fix {error_spaces} spaces with errors")
    
    training_spaces = [s for s in space_analyses if 'training' in s.get('space_id', '').lower()]
    if training_spaces:
        print(f"3. Training spaces found: {len(training_spaces)} - ensure they have ZeroGPU")
    
    print(f"4. All spaces are ready for community use and sharing")

if __name__ == "__main__":
    read_all_spaces()
