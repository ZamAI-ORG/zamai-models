#!/usr/bin/env python3
"""
Update all HuggingFace Spaces with ZeroGPU support and improvements
"""

import os
from pathlib import Path
from huggingface_hub import HfApi
import tempfile

def read_hf_token():
    token_file = Path('/workspaces/ZamAI-Pro-Models/HF-Token.txt')
    return token_file.read_text().strip()

def update_space_with_zerogpu(space_id, token):
    """Update a space with ZeroGPU support"""
    api = HfApi(token=token)
    
    print(f"\n🔧 Updating {space_id}...")
    
    try:
        # Try to download and update app.py
        try:
            from huggingface_hub import hf_hub_download
            app_path = hf_hub_download(
                repo_id=space_id,
                filename="app.py",
                repo_type="space",
                token=token
            )
            
            with open(app_path, 'r') as f:
                content = f.read()
            
            # Check current state
            has_spaces_import = "import spaces" in content
            has_gpu_decorator = "@spaces.GPU" in content
            
            print(f"   📄 Current state:")
            print(f"      - Has 'import spaces': {has_spaces_import}")
            print(f"      - Has '@spaces.GPU': {has_gpu_decorator}")
            
            # Update content if needed
            if not has_spaces_import or not has_gpu_decorator:
                updated_content = add_zerogpu_support(content, has_spaces_import, has_gpu_decorator)
                
                # Create temp file and upload
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                    tmp.write(updated_content)
                    tmp_path = tmp.name
                
                api.upload_file(
                    path_or_fileobj=tmp_path,
                    path_in_repo="app.py",
                    repo_id=space_id,
                    repo_type="space",
                    commit_message="🚀 Add ZeroGPU support for better performance"
                )
                
                os.unlink(tmp_path)
                print(f"   ✅ Updated app.py with ZeroGPU support")
            else:
                print(f"   ✅ app.py already has ZeroGPU support")
                
        except Exception as e:
            print(f"   ⚠️  Could not update app.py: {e}")
        
        # Update requirements.txt
        try:
            req_path = hf_hub_download(
                repo_id=space_id,
                filename="requirements.txt",
                repo_type="space",
                token=token
            )
            
            with open(req_path, 'r') as f:
                req_content = f.read()
            
            if "spaces" not in req_content.lower():
                # Add spaces package
                if not req_content.endswith('\n'):
                    req_content += '\n'
                req_content += "spaces\n"
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                    tmp.write(req_content)
                    tmp_path = tmp.name
                
                api.upload_file(
                    path_or_fileobj=tmp_path,
                    path_in_repo="requirements.txt",
                    repo_id=space_id,
                    repo_type="space",
                    commit_message="📦 Add spaces package for ZeroGPU"
                )
                
                os.unlink(tmp_path)
                print(f"   ✅ Updated requirements.txt")
            else:
                print(f"   ✅ requirements.txt already has spaces")
                
        except Exception as e:
            print(f"   ⚠️  Could not update requirements.txt: {e}")
        
        # Update README with usage info
        try:
            update_space_readme(api, space_id)
        except Exception as e:
            print(f"   ⚠️  Could not update README: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error updating space: {e}")
        return False

def add_zerogpu_support(content, has_spaces_import, has_gpu_decorator):
    """Add ZeroGPU support to app.py content"""
    lines = content.split('\n')
    result = []
    
    # Add spaces import if missing
    if not has_spaces_import:
        # Find the best place to add import (after other imports)
        import_added = False
        for i, line in enumerate(lines):
            result.append(line)
            
            # After import statements, add spaces
            if not import_added and line.strip().startswith(('import ', 'from ')):
                # Look ahead to see if next line is also an import
                if i + 1 >= len(lines) or not lines[i + 1].strip().startswith(('import ', 'from ')):
                    result.append('import spaces')
                    import_added = True
        
        if not import_added:
            # Add at the top if no imports found
            result.insert(0, 'import spaces')
            result.insert(1, '')
        
        lines = result
        result = []
    
    # Add @spaces.GPU decorator if missing
    if not has_gpu_decorator:
        for i, line in enumerate(lines):
            # Look for function definitions that should use GPU
            if line.strip().startswith('def ') and any(keyword in line.lower() for keyword in 
                ['predict', 'generate', 'inference', 'process', 'transcribe', 'translate']):
                result.append('@spaces.GPU')
            result.append(line)
    else:
        result = lines
    
    return '\n'.join(result)

def update_space_readme(api, space_id):
    """Update space README with usage information"""
    from huggingface_hub import hf_hub_download
    
    try:
        readme_path = hf_hub_download(
            repo_id=space_id,
            filename="README.md",
            repo_type="space",
            token=api.token
        )
        
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        
        # Check if already has performance badge
        if "ZeroGPU" not in readme_content:
            # Add ZeroGPU badge
            badge = "\n![ZeroGPU](https://img.shields.io/badge/ZeroGPU-Enabled-blue)\n"
            
            # Add after title or at the beginning
            if readme_content.startswith('---'):
                # Has frontmatter, add after it
                parts = readme_content.split('---', 2)
                if len(parts) >= 3:
                    readme_content = f"---{parts[1]}---{badge}{parts[2]}"
            else:
                readme_content = badge + readme_content
            
            # Upload updated README
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
                tmp.write(readme_content)
                tmp_path = tmp.name
            
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="README.md",
                repo_id=space_id,
                repo_type="space",
                commit_message="📝 Update README with ZeroGPU badge"
            )
            
            os.unlink(tmp_path)
            print(f"   ✅ Updated README.md")
    
    except Exception:
        # README might not exist, that's okay
        pass

def main():
    print("🚀 Updating All HuggingFace Spaces")
    print("=" * 70)
    
    token = read_hf_token()
    api = HfApi(token=token)
    
    # Get all spaces
    print("\n🔍 Fetching spaces...")
    spaces = list(api.list_spaces(author='tasal9'))
    
    print(f"   Found {len(spaces)} spaces to update\n")
    
    # Update each space
    updated = 0
    failed = 0
    
    for space in spaces:
        success = update_space_with_zerogpu(space.id, token)
        if success:
            updated += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 UPDATE SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully updated: {updated}")
    print(f"❌ Failed: {failed}")
    print(f"📝 Total spaces: {len(spaces)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
