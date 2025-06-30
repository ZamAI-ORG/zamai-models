#!/usr/bin/env python3
"""
Analyze ZamAI dataset
Wrapper script for easy execution
"""

import subprocess
import sys
import os

def main():
    print("🇦🇫 Analyze ZamAI dataset")
    print("=" * 50)
    
    script_path = "scripts/analysis/analyze_zamai_dataset.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return 1
    
    try:
        # Run the target script
        result = subprocess.run([sys.executable, script_path], 
                              cwd=os.getcwd(),
                              check=True)
        print(f"\n✅ Completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        return 1

if __name__ == "__main__":
    exit(main())
