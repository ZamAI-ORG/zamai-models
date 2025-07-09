#!/usr/bin/env python3
"""
ZeroGPU Space Validation
Check if all files are ready for upload
"""

from pathlib import Path
import json

def validate_spaces():
    """Validate all ZeroGPU training space files"""
    
    zerogpu_dir = Path("/workspaces/ZamAI-Pro-Models/zerogpu_files")
    
    if not zerogpu_dir.exists():
        print("❌ zerogpu_files directory not found!")
        return
    
    print("🔍 Validating ZeroGPU Training Spaces")
    print("=" * 50)
    
    required_files = ["app.py", "requirements.txt", "README.md"]
    valid_spaces = 0
    
    for space_dir in zerogpu_dir.iterdir():
        if space_dir.is_dir():
            print(f"\n📁 {space_dir.name}")
            
            all_files_present = True
            for req_file in required_files:
                file_path = space_dir / req_file
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    print(f"  ✅ {req_file} ({file_size} bytes)")
                else:
                    print(f"  ❌ {req_file} - Missing!")
                    all_files_present = False
            
            # Check README.md for proper YAML front matter
            readme_path = space_dir / "README.md"
            if readme_path.exists():
                content = readme_path.read_text()
                if content.startswith("---") and "hardware: zero-a10g" in content:
                    print(f"  ✅ README.md has proper Space config")
                else:
                    print(f"  ⚠️  README.md missing Space config")
            
            # Check app.py for @spaces.GPU decorator
            app_path = space_dir / "app.py"
            if app_path.exists():
                content = app_path.read_text()
                if "@spaces.GPU" in content and "import spaces" in content:
                    print(f"  ✅ app.py has ZeroGPU integration")
                else:
                    print(f"  ⚠️  app.py missing ZeroGPU decorators")
            
            if all_files_present:
                valid_spaces += 1
                print(f"  🎉 {space_dir.name} is ready for upload!")
    
    print(f"\n📊 Summary: {valid_spaces} spaces ready for deployment")
    
    if valid_spaces > 0:
        print(f"\n🚀 Next Steps:")
        print(f"1. Go to https://huggingface.co/new-space")
        print(f"2. Create a space for each directory")
        print(f"3. Set Hardware to 'ZeroGPU - A10G'")
        print(f"4. Upload the 3 files from each directory")
        print(f"5. Wait for build and start training!")
    else:
        print(f"\n❌ No valid spaces found. Run setup_files.py first.")

if __name__ == "__main__":
    validate_spaces()
