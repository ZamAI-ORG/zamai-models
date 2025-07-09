#!/usr/bin/env python3
"""
Fix ZeroGPU issues in existing spaces
"""

import os
from huggingface_hub import HfApi, upload_file
import json

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def fix_space_zerogpu_issues(space_id, token):
    """Fix ZeroGPU issues in a space"""
    api = HfApi(token=token)
    
    print(f"\n🔧 Fixing {space_id}...")
    
    try:
        # Download current app.py
        try:
            app_content = api.hf_hub_download(
                repo_id=space_id,
                filename="app.py",
                repo_type="space"
            )
            
            with open(app_content, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"   ❌ Could not download app.py: {e}")
            return False
        
        # Check if already has spaces import
        if "import spaces" in content:
            print("   ✅ Already has 'import spaces'")
            has_spaces_import = True
        else:
            print("   🔄 Adding 'import spaces'")
            has_spaces_import = False
        
        # Check if already has @spaces.GPU decorator
        if "@spaces.GPU" in content:
            print("   ✅ Already has '@spaces.GPU' decorator")
            has_gpu_decorator = True
        else:
            print("   🔄 Adding '@spaces.GPU' decorator")
            has_gpu_decorator = False
        
        # Fix the content if needed
        if not has_spaces_import or not has_gpu_decorator:
            fixed_content = fix_app_py_content(content, has_spaces_import, has_gpu_decorator)
            
            # Save fixed content temporarily
            temp_file = f"/tmp/{space_id.replace('/', '_')}_app.py"
            with open(temp_file, 'w') as f:
                f.write(fixed_content)
            
            # Upload fixed app.py
            api.upload_file(
                path_or_fileobj=temp_file,
                path_in_repo="app.py",
                repo_id=space_id,
                repo_type="space"
            )
            print("   ✅ Fixed app.py")
            
            # Clean up
            os.remove(temp_file)
        
        # Fix requirements.txt
        try:
            requirements_content = api.hf_hub_download(
                repo_id=space_id,
                filename="requirements.txt",
                repo_type="space"
            )
            
            with open(requirements_content, 'r') as f:
                req_content = f.read()
            
            if "spaces" not in req_content:
                print("   🔄 Adding 'spaces' to requirements.txt")
                
                # Add spaces to requirements
                if req_content and not req_content.endswith('\n'):
                    req_content += '\n'
                req_content += "spaces\n"
                
                # Save fixed requirements temporarily
                temp_req_file = f"/tmp/{space_id.replace('/', '_')}_requirements.txt"
                with open(temp_req_file, 'w') as f:
                    f.write(req_content)
                
                # Upload fixed requirements.txt
                api.upload_file(
                    path_or_fileobj=temp_req_file,
                    path_in_repo="requirements.txt",
                    repo_id=space_id,
                    repo_type="space"
                )
                print("   ✅ Fixed requirements.txt")
                
                # Clean up
                os.remove(temp_req_file)
            else:
                print("   ✅ requirements.txt already has 'spaces'")
                
        except Exception as e:
            print(f"   ⚠️  Could not fix requirements.txt: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error fixing space: {e}")
        return False

def fix_app_py_content(content, has_spaces_import, has_gpu_decorator):
    """Fix app.py content for ZeroGPU support"""
    lines = content.split('\n')
    fixed_lines = []
    
    # Add spaces import if missing
    if not has_spaces_import:
        # Add after other imports
        import_added = False
        for i, line in enumerate(lines):
            fixed_lines.append(line)
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                # Add spaces import after the last import
                if i + 1 >= len(lines) or not (lines[i + 1].strip().startswith('import ') or lines[i + 1].strip().startswith('from ')):
                    if not import_added:
                        fixed_lines.append('import spaces')
                        import_added = True
        
        if not import_added:
            # If no imports found, add at the beginning
            fixed_lines.insert(0, 'import spaces')
        
        lines = fixed_lines
        fixed_lines = []
    
    # Add @spaces.GPU decorator if missing
    if not has_gpu_decorator:
        for i, line in enumerate(lines):
            # Look for function definitions that might need GPU
            if line.strip().startswith('def ') and ('predict' in line or 'generate' in line or 'inference' in line or 'process' in line):
                fixed_lines.append('@spaces.GPU')
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
    else:
        fixed_lines = lines
    
    return '\n'.join(fixed_lines)

def main():
    """Fix ZeroGPU issues in all problematic spaces"""
    print("🔧 Fixing ZeroGPU Issues in Spaces")
    print("=" * 60)
    
    # Read the analysis results
    with open('/workspaces/ZamAI-Pro-Models/space_analysis.json', 'r') as f:
        analysis = json.load(f)
    
    token = read_hf_token()
    
    # Get spaces with issues from the analysis results
    spaces_with_issues = []
    
    # Check the analyses section
    for space_info in analysis['analyses']:
        if 'issues' in space_info and space_info['issues']:
            spaces_with_issues.append(space_info['space_id'])
    
    print(f"🎯 Found {len(spaces_with_issues)} spaces to fix:")
    for space_id in spaces_with_issues:
        print(f"   - {space_id}")
    
    # Fix each space
    fixed_count = 0
    for space_id in spaces_with_issues:
        success = fix_space_zerogpu_issues(space_id, token)
        if success:
            fixed_count += 1
    
    print(f"\n📈 SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully fixed: {fixed_count}/{len(spaces_with_issues)} spaces")
    
    if fixed_count < len(spaces_with_issues):
        print(f"❌ Failed to fix: {len(spaces_with_issues) - fixed_count} spaces")

if __name__ == "__main__":
    main()
