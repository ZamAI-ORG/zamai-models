#!/usr/bin/env python3
"""
Quick verification of newly created spaces
"""

from huggingface_hub import HfApi, list_spaces

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def check_new_spaces():
    """Check what spaces now exist"""
    token = read_hf_token()
    
    print("🔍 Checking current spaces...")
    
    # Get all current spaces
    spaces = list(list_spaces(author="tasal9", token=token))
    space_names = [space.id for space in spaces]
    
    print(f"\\n🚀 Found {len(space_names)} total spaces:")
    
    # Categorize spaces
    testing_spaces = [s for s in space_names if 'testing' in s]
    training_spaces = [s for s in space_names if 'training' in s]
    other_spaces = [s for s in space_names if 'testing' not in s and 'training' not in s]
    
    print(f"\\n📝 Testing Spaces ({len(testing_spaces)}):")
    for space in sorted(testing_spaces):
        print(f"   - {space}")
    
    print(f"\\n🏋️  Training Spaces ({len(training_spaces)}):")
    for space in sorted(training_spaces):
        print(f"   - {space}")
    
    print(f"\\n❓ Other Spaces ({len(other_spaces)}):")
    for space in sorted(other_spaces):
        print(f"   - {space}")
    
    # Check for specific expected spaces
    expected_testing = [
        "tasal9/pashto-base-bloom-testing",
        "tasal9/Multilingual-ZamAI-Embeddings-testing",
        "tasal9/ZamAI-Mistral-7B-Pashto-testing",
        "tasal9/ZamAI-Phi-3-Mini-Pashto-testing",
        "tasal9/ZamAI-Whisper-v3-Pashto-testing"
    ]
    
    expected_training = [
        "tasal9/pashto-base-bloom-training",
        "tasal9/ZamAI-LIama3-Pashto-training",
        "tasal9/Multilingual-ZamAI-Embeddings-training", 
        "tasal9/ZamAI-Mistral-7B-Pashto-training",
        "tasal9/ZamAI-Phi-3-Mini-Pashto-training",
        "tasal9/ZamAI-Whisper-v3-Pashto-training"
    ]
    
    print(f"\\n✅ VERIFICATION")
    print("=" * 50)
    
    missing_testing = [s for s in expected_testing if s not in space_names]
    missing_training = [s for s in expected_training if s not in space_names]
    
    print(f"📝 Expected testing spaces: {len(expected_testing)}")
    print(f"📝 Found testing spaces: {len([s for s in expected_testing if s in space_names])}")
    print(f"📝 Missing testing spaces: {len(missing_testing)}")
    
    print(f"🏋️  Expected training spaces: {len(expected_training)}")
    print(f"🏋️  Found training spaces: {len([s for s in expected_training if s in space_names])}")
    print(f"🏋️  Missing training spaces: {len(missing_training)}")
    
    if missing_testing:
        print(f"\\n❌ Missing testing spaces:")
        for space in missing_testing:
            print(f"   - {space}")
    
    if missing_training:
        print(f"\\n❌ Missing training spaces:")
        for space in missing_training:
            print(f"   - {space}")
    
    if not missing_testing and not missing_training:
        print(f"\\n🎉 All expected spaces created successfully!")
    
    return len(space_names), len(missing_testing), len(missing_training)

if __name__ == "__main__":
    check_new_spaces()
