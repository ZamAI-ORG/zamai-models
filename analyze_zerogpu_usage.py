#!/usr/bin/env python3
"""
Analyze ZeroGPU usage and optimize space allocation
"""

from huggingface_hub import HfApi, list_spaces
import json

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def analyze_zerogpu_usage():
    """Analyze current ZeroGPU usage"""
    token = read_hf_token()
    
    print("🔍 Analyzing ZeroGPU Usage")
    print("=" * 60)
    
    # Get all spaces
    spaces = list(list_spaces(author="tasal9", token=token))
    
    zerogpu_spaces = []
    cpu_spaces = []
    
    for space in spaces:
        # Check if space uses ZeroGPU (this is based on the hardware field in README)
        try:
            # Note: We need to check the space configuration
            # For now, let's categorize based on naming patterns and known info
            space_id = space.id
            
            # These are likely ZeroGPU spaces based on our recent work
            if any(keyword in space_id for keyword in ['testing', 'training', 'bloom', 'whisper', 'embeddings', 'llama', 'mistral', 'phi']):
                zerogpu_spaces.append(space_id)
            else:
                cpu_spaces.append(space_id)
                
        except Exception as e:
            print(f"   ⚠️  Could not analyze {space.id}: {e}")
    
    print(f"\\n📊 Current Space Distribution:")
    print(f"🔥 ZeroGPU Spaces: {len(zerogpu_spaces)}")
    print(f"💻 CPU Spaces: {len(cpu_spaces)}")
    print(f"📦 Total Spaces: {len(spaces)}")
    
    print(f"\\n🔥 ZeroGPU Spaces ({len(zerogpu_spaces)}):")
    for space in sorted(zerogpu_spaces):
        print(f"   - {space}")
    
    print(f"\\n💻 CPU Spaces ({len(cpu_spaces)}):")
    for space in sorted(cpu_spaces):
        print(f"   - {space}")
    
    # Prioritization recommendations
    print(f"\\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    
    # Priority models that MUST have ZeroGPU
    priority_models = {
        "tasal9/ZamAI-LIama3-Pashto": "Large language model - needs GPU",
        "tasal9/ZamAI-Mistral-7B-Pashto": "7B parameter model - needs GPU", 
        "tasal9/ZamAI-Whisper-v3-Pashto": "Speech processing - benefits from GPU",
        "tasal9/Multilingual-ZamAI-Embeddings": "Embedding computation - benefits from GPU",
        "tasal9/pashto-base-bloom": "BLOOM model - needs GPU",
        "tasal9/ZamAI-Phi-3-Mini-Pashto": "Phi-3 model - benefits from GPU"
    }
    
    essential_spaces = []
    optional_zerogpu = []
    
    for space in zerogpu_spaces:
        is_essential = False
        for model_id, reason in priority_models.items():
            model_name = model_id.split('/')[-1]
            if model_name in space:
                essential_spaces.append((space, reason))
                is_essential = True
                break
        
        if not is_essential:
            optional_zerogpu.append(space)
    
    print(f"⭐ Essential ZeroGPU Spaces ({len(essential_spaces)}):")
    for space, reason in essential_spaces:
        print(f"   - {space}: {reason}")
    
    print(f"\\n❓ Optional ZeroGPU Spaces ({len(optional_zerogpu)}):")
    for space in optional_zerogpu:
        print(f"   - {space}: Could potentially use CPU")
    
    # Recommend actions
    print(f"\\n🎯 RECOMMENDED ACTIONS")
    print("-" * 30)
    
    current_zerogpu_count = len(zerogpu_spaces)
    zerogpu_limit = 10
    available_slots = max(0, zerogpu_limit - current_zerogpu_count)
    
    print(f"1. Current ZeroGPU usage: {current_zerogpu_count}/{zerogpu_limit}")
    print(f"2. Available slots: {available_slots}")
    
    if available_slots == 0:
        print(f"3. 🚨 At ZeroGPU limit! Need to optimize:")
        print(f"   • Move {len(optional_zerogpu)} optional spaces to CPU")
        print(f"   • Focus on {len(essential_spaces)} essential spaces")
        print(f"   • Consider consolidating similar functionality")
    else:
        print(f"3. ✅ Can create {available_slots} more ZeroGPU spaces")
    
    # Create optimization plan
    optimization_plan = {
        "current_zerogpu_count": current_zerogpu_count,
        "zerogpu_limit": zerogpu_limit,
        "available_slots": available_slots,
        "essential_spaces": [space for space, _ in essential_spaces],
        "optional_spaces": optional_zerogpu,
        "recommendations": []
    }
    
    if available_slots == 0:
        # Need to free up slots
        spaces_to_move = min(len(optional_zerogpu), 4)  # Move 4 to free up slots
        optimization_plan["recommendations"] = [
            f"Move {spaces_to_move} optional spaces to CPU hardware",
            "Prioritize creating testing spaces for main models",
            "Create 1-2 consolidated training spaces instead of individual ones",
            "Consider creating multi-model demo spaces"
        ]
    
    # Save optimization plan
    with open('/workspaces/ZamAI-Pro-Models/zerogpu_optimization.json', 'w') as f:
        json.dump(optimization_plan, f, indent=2)
    
    print(f"\\n💾 Optimization plan saved to: zerogpu_optimization.json")
    
    return optimization_plan

if __name__ == "__main__":
    analyze_zerogpu_usage()
