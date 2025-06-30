#!/usr/bin/env python3
"""
ZamAI Quick Commands
Easy access to all ZamAI functionality
"""

import os
import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("🇦🇫 ZamAI Quick Commands")
        print("=" * 40)
        print("Usage: python zamai.py <command>")
        print()
        print("📚 Available Commands:")
        print("  setup      - Validate setup and environment")
        print("  analyze    - Analyze ZamAI dataset")
        print("  test       - Test existing models")
        print("  train      - Train ZamAI V4 model")
        print("  status     - Check model status")
        print("  help       - Show this help")
        print()
        print("📁 Direct Script Access:")
        print("  scripts/training/train_zamai_v4.py")
        print("  scripts/analysis/analyze_zamai_dataset.py")
        print("  scripts/testing/test_models.py")
        print("  scripts/utils/validate_setup.py")
        return

    command = sys.argv[1].lower()
    
    script_map = {
        'setup': 'scripts/utils/validate_setup.py',
        'validate': 'scripts/utils/validate_setup.py',
        'analyze': 'scripts/analysis/analyze_zamai_dataset.py',
        'dataset': 'scripts/analysis/analyze_zamai_dataset.py',
        'test': 'scripts/testing/test_models.py',
        'train': 'scripts/training/train_zamai_v4.py',
        'status': 'scripts/utils/check_dataset_access.py',
        'check': 'scripts/utils/basic_check.py'
    }
    
    if command == 'help':
        main()
        return
    
    if command in script_map:
        script_path = script_map[command]
        print(f"🚀 Running: {script_path}")
        print("-" * 40)
        
        try:
            subprocess.run([sys.executable, script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running script: {e}")
        except FileNotFoundError:
            print(f"❌ Script not found: {script_path}")
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python zamai.py help' for available commands")

if __name__ == "__main__":
    main()
