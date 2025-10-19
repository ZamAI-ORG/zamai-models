#!/usr/bin/env python3
"""
Fix Specific Space Runtime Errors
Diagnose and fix individual spaces with targeted solutions
"""

import sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import tempfile
import os

def read_hf_token():
    return Path('/workspaces/ZamAI-Pro-Models/HF-Token.txt').read_text().strip()

def get_space_error_logs(api, space_id):
    """Get error information from space"""
    try:
        info = api.space_info(space_id)
        print(f"\n📋 Space Info for {space_id}:")
        print(f"   SDK: {info.sdk if hasattr(info, 'sdk') else 'Unknown'}")
        print(f"   Runtime: {info.runtime.stage if hasattr(info, 'runtime') else 'Unknown'}")
        print(f"   Hardware: {info.runtime.hardware if hasattr(info, 'runtime') and hasattr(info.runtime, 'hardware') else 'None'}")
        return info
    except Exception as e:
        print(f"   ❌ Error getting space info: {e}")
        return None

def fix_gradio_space(api, space_id, token):
    """Fix common Gradio space issues"""
    print(f"\n🔧 Fixing Gradio Space: {space_id}")
    
    fixes_applied = []
    
    # 1. Check and fix requirements.txt
    try:
        req_path = hf_hub_download(
            repo_id=space_id,
            filename="requirements.txt",
            repo_type="space",
            token=token
        )
        
        with open(req_path, 'r') as f:
            requirements = f.read()
        
        print(f"   📦 Current requirements:\n{requirements[:200]}...")
        
        # Ensure essential packages
        essential_packages = ['gradio', 'torch', 'transformers']
        missing_packages = []
        
        for pkg in essential_packages:
            if pkg not in requirements.lower():
                missing_packages.append(pkg)
        
        if missing_packages:
            print(f"   ⚠️  Missing packages: {missing_packages}")
            
            # Add missing packages
            new_requirements = requirements
            if not new_requirements.endswith('\n'):
                new_requirements += '\n'
            
            for pkg in missing_packages:
                if pkg == 'gradio':
                    new_requirements += 'gradio>=4.0.0\n'
                elif pkg == 'torch':
                    new_requirements += 'torch>=2.0.0\n'
                elif pkg == 'transformers':
                    new_requirements += 'transformers>=4.30.0\n'
            
            # Upload fixed requirements
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                tmp.write(new_requirements)
                tmp_path = tmp.name
            
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="requirements.txt",
                repo_id=space_id,
                repo_type="space",
                commit_message="🔧 Fix: Add missing dependencies"
            )
            
            os.unlink(tmp_path)
            fixes_applied.append("Updated requirements.txt")
            print(f"   ✅ Updated requirements.txt")
    
    except Exception as e:
        print(f"   ⚠️  Could not fix requirements.txt: {e}")
    
    # 2. Check and create/fix README.md with proper config
    try:
        # Create proper README with YAML frontmatter
        readme_content = f"""---
title: {space_id.split('/')[-1]}
emoji: 🇦🇫
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - pashto
  - afghanistan
  - zamai
  - ai
---

# {space_id.split('/')[-1]}

ZamAI Pashto AI Model Demo

This space demonstrates the capabilities of the {space_id} model for Pashto language processing.

## Features
- Pashto language support
- Real-time inference
- User-friendly interface

## Usage
Simply enter your text and click submit to see the model in action!

---

🇦🇫 Built with ❤️ for the Afghan community
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
            tmp.write(readme_content)
            tmp_path = tmp.name
        
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=space_id,
            repo_type="space",
            commit_message="📝 Fix: Update README with proper configuration"
        )
        
        os.unlink(tmp_path)
        fixes_applied.append("Updated README.md")
        print(f"   ✅ Updated README.md with proper config")
    
    except Exception as e:
        print(f"   ⚠️  Could not update README.md: {e}")
    
    # 3. Check app.py exists and has basic structure
    try:
        app_path = hf_hub_download(
            repo_id=space_id,
            filename="app.py",
            repo_type="space",
            token=token
        )
        
        with open(app_path, 'r') as f:
            app_content = f.read()
        
        # Check for common issues
        issues = []
        if 'import gradio' not in app_content:
            issues.append("Missing 'import gradio'")
        if '.launch()' not in app_content and 'launch()' not in app_content:
            issues.append("Missing '.launch()' call")
        
        if issues:
            print(f"   ⚠️  App.py issues found: {issues}")
            print(f"   💡 Manual review recommended")
        else:
            print(f"   ✅ app.py structure looks good")
    
    except Exception as e:
        print(f"   ❌ Could not check app.py: {e}")
    
    return fixes_applied

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_space_runtime.py <space-id>")
        print("\nExample: python fix_space_runtime.py tasal9/pashto-base-bloom-space")
        sys.exit(1)
    
    space_id = sys.argv[1]
    
    print(f"🔍 Diagnosing and Fixing Space: {space_id}")
    print("=" * 70)
    
    token = read_hf_token()
    api = HfApi(token=token)
    
    # Get current state
    space_info = get_space_error_logs(api, space_id)
    
    if not space_info:
        print("\n❌ Could not retrieve space information")
        sys.exit(1)
    
    # Apply fixes
    fixes = fix_gradio_space(api, space_id, token)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    if fixes:
        print(f"✅ Applied {len(fixes)} fixes:")
        for fix in fixes:
            print(f"   - {fix}")
        print("\n🔄 Space is rebuilding. Check status at:")
        print(f"   https://huggingface.co/spaces/{space_id}")
    else:
        print("⚠️  No automatic fixes could be applied")
        print("\n💡 Recommended manual actions:")
        print("   1. Check space logs for specific errors")
        print("   2. Verify model loading code in app.py")
        print("   3. Ensure hardware allocation if needed")
        print("   4. Test locally before deploying")
    print("=" * 70)

if __name__ == "__main__":
    main()
