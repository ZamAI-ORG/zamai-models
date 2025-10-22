#!/usr/bin/env python3
"""Quick script to check HuggingFace Spaces status"""

from huggingface_hub import HfApi
from pathlib import Path

def main():
    token = Path('/workspaces/ZamAI-Pro-Models/HF-Token.txt').read_text().strip()
    api = HfApi(token=token)
    
    spaces_to_check = [
        'pashto-base-bloom-space',
        'ZamAI-Mistral-7B-Pashto-space',
        'ZamAI-Pashto-Translator-FacebookNLB-ps-en',
        'ZamAI-mt5-Pashto-training-Space'
    ]
    
    print('🔍 Checking HuggingFace Spaces Status')
    print('=' * 70)
    
    for space_name in spaces_to_check:
        space_id = f'tasal9/{space_name}'
        try:
            info = api.space_info(space_id)
            status = info.runtime.stage if info.runtime else 'Unknown'
            sdk = info.sdk if hasattr(info, 'sdk') else 'Unknown'
            
            status_icon = '✅' if status == 'RUNNING' else '🔄' if status == 'APP_STARTING' else '❌'
            
            print(f'\n{status_icon} {space_name}')
            print(f'   Status: {status}')
            print(f'   SDK: {sdk}')
            print(f'   URL: https://huggingface.co/spaces/{space_id}')
            
        except Exception as e:
            print(f'\n❌ {space_name}')
            print(f'   ERROR: {str(e)[:100]}')
    
    print('\n' + '=' * 70)

if __name__ == '__main__':
    main()
