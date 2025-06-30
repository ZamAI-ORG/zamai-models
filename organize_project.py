#!/usr/bin/env python3
"""
ZamAI Setup and Path Fixer
Fixes all import paths and updates script references after reorganization
"""

import os
import re

def fix_script_paths():
    """Fix all script paths after reorganization"""
    print("🔧 Fixing script paths after reorganization...")
    
    # Update train_zamai_v4.py to use correct paths
    train_script_path = "scripts/training/train_zamai_v4.py"
    if os.path.exists(train_script_path):
        with open(train_script_path, 'r') as f:
            content = f.read()
        
        # Update config path
        content = content.replace(
            'config_path = "fine-tuning/configs/pashto_chat_config.json"',
            'config_path = "../../configs/pashto_chat_config.json"'
        )
        
        # Update sys.path for imports
        content = content.replace(
            'os.chdir("fine-tuning")',
            'os.chdir("../../")'
        )
        
        # Update import path
        content = content.replace(
            'from train_pashto_chat import PashtoModelTrainer',
            'from scripts.training.train_pashto_chat import PashtoModelTrainer'
        )
        
        with open(train_script_path, 'w') as f:
            f.write(content)
        
        print(f"  ✅ Fixed: {train_script_path}")
    
    # Update validate_setup.py paths
    validate_script_path = "scripts/utils/validate_setup.py"
    if os.path.exists(validate_script_path):
        with open(validate_script_path, 'r') as f:
            content = f.read()
        
        # Update file paths to work from new structure
        updates = [
            ('HF-Token.txt', '../../HF-Token.txt'),
            ('fine-tuning/train_pashto_chat.py', '../training/train_pashto_chat.py'),
            ('fine-tuning/configs/pashto_chat_config.json', '../../configs/pashto_chat_config.json'),
            ('train_zamai_v4.py', '../training/train_zamai_v4.py'),
            ('analyze_zamai_dataset.py', '../analysis/analyze_zamai_dataset.py'),
            ('test_models.py', '../testing/test_models.py')
        ]
        
        for old_path, new_path in updates:
            content = content.replace(f'"{old_path}"', f'"{new_path}"')
        
        with open(validate_script_path, 'w') as f:
            f.write(content)
        
        print(f"  ✅ Fixed: {validate_script_path}")
    
    # Update zamai.py script paths
    zamai_script_path = "zamai.py"
    if os.path.exists(zamai_script_path):
        print(f"  ✅ ZamAI quick commands ready: {zamai_script_path}")

def create_run_scripts():
    """Create easy-to-run scripts in the root directory"""
    
    scripts = {
        "run_setup.py": {
            "description": "Validate ZamAI setup",
            "script": "scripts/utils/validate_setup.py"
        },
        "run_analysis.py": {
            "description": "Analyze ZamAI dataset", 
            "script": "scripts/analysis/analyze_zamai_dataset.py"
        },
        "run_training.py": {
            "description": "Train ZamAI V4 model",
            "script": "scripts/training/train_zamai_v4.py"
        },
        "run_testing.py": {
            "description": "Test existing models",
            "script": "scripts/testing/test_models.py"
        }
    }
    
    for script_name, info in scripts.items():
        script_content = f'''#!/usr/bin/env python3
"""
{info["description"]}
Wrapper script for easy execution
"""

import subprocess
import sys
import os

def main():
    print("🇦🇫 {info['description']}")
    print("=" * 50)
    
    script_path = "{info['script']}"
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {{script_path}}")
        return 1
    
    try:
        # Run the target script
        result = subprocess.run([sys.executable, script_path], 
                              cwd=os.getcwd(),
                              check=True)
        print(f"\\n✅ Completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Error: {{e}}")
        return 1
    except KeyboardInterrupt:
        print(f"\\n⚠️  Interrupted by user")
        return 1

if __name__ == "__main__":
    exit(main())
'''
        
        with open(script_name, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_name, 0o755)
        print(f"  ✅ Created: {script_name}")

def main():
    print("🇦🇫 ZamAI Project Reorganization")
    print("=" * 50)
    
    # Fix script paths
    fix_script_paths()
    
    # Create wrapper scripts
    print(f"\n📝 Creating wrapper scripts...")
    create_run_scripts()
    
    print(f"\n✅ Project reorganization complete!")
    print(f"\n🚀 Quick Start Commands:")
    print(f"  python run_setup.py      # Validate setup")
    print(f"  python run_analysis.py   # Analyze dataset") 
    print(f"  python run_training.py   # Train model")
    print(f"  python run_testing.py    # Test models")
    print(f"  python zamai.py <command> # Quick commands")
    
    print(f"\n📁 Organized Structure:")
    print(f"  scripts/training/   # Training scripts")
    print(f"  scripts/testing/    # Testing scripts") 
    print(f"  scripts/analysis/   # Analysis scripts")
    print(f"  scripts/utils/      # Utility scripts")
    print(f"  configs/           # Configuration files")
    print(f"  data/processed/    # Results and processed data")
    print(f"  outputs/           # Training outputs")
    print(f"  logs/              # Training logs")

if __name__ == "__main__":
    main()
