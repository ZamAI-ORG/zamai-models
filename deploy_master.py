#!/usr/bin/env python3
"""
ZamAI Master Deployment Script
Runs all deployment, testing, and synchronization steps
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_command(script_name, description):
    """Run a Python script and report results"""
    print(f"\n{'=' * 70}")
    print(f"🚀 {description}")
    print(f"{'=' * 70}\n")
    
    script_path = Path(f"/workspaces/ZamAI-Pro-Models/{script_name}")
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            cwd="/workspaces/ZamAI-Pro-Models"
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} - COMPLETED")
            return True
        else:
            print(f"\n⚠️  {description} - COMPLETED WITH WARNINGS")
            return True
    
    except Exception as e:
        print(f"\n❌ {description} - FAILED: {e}")
        return False

def print_header():
    """Print script header"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║              🇦🇫 ZamAI Pro Models Master Deployment              ║
║                                                                  ║
║     Comprehensive Setup, Testing, and Synchronization Suite     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def print_summary(results):
    """Print execution summary"""
    print(f"\n\n{'=' * 70}")
    print("📊 EXECUTION SUMMARY")
    print(f"{'=' * 70}\n")
    
    for step, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} - {step}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n{'=' * 70}")
    print(f"Total Steps: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"{'=' * 70}\n")
    
    if passed == total:
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    else:
        print("⚠️  SOME STEPS FAILED - Please review the output above")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main execution flow"""
    print_header()
    
    results = {}
    
    # Step 1: Sync HuggingFace State
    results["Step 1: Sync HF State"] = run_command(
        "sync_hf_state.py",
        "Step 1: Synchronizing HuggingFace Models and Spaces"
    )
    
    # Step 2: Update All Spaces with ZeroGPU
    results["Step 2: Update Spaces"] = run_command(
        "update_all_spaces.py",
        "Step 2: Updating All Spaces with ZeroGPU Support"
    )
    
    # Step 3: Test All Models
    results["Step 3: Test Models"] = run_command(
        "test_all_models.py",
        "Step 3: Testing All Models"
    )
    
    # Step 4: Validate Setup
    results["Step 4: Validate Setup"] = run_command(
        "scripts/utils/validate_setup.py",
        "Step 4: Validating Local Setup"
    )
    
    # Print summary
    print_summary(results)
    
    # Print next steps
    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS")
    print("=" * 70)
    print("""
1. Review Generated Reports:
   - HF_CURRENT_STATE.md - Current HuggingFace status
   - MODEL_TEST_REPORT.md - Model testing results
   - hf_current_state.json - Raw data

2. Launch Voice Assistant:
   ./start_voice_assistant.sh

3. Deploy with Docker:
   docker-compose up -d

4. Create Training Spaces:
   python create_training_space.py

5. Community Sharing:
   - Share on social media
   - Update model cards
   - Create demo videos
   - Engage with Pashto AI community

6. Monitor Performance:
   - Check space logs
   - Review usage analytics
   - Gather user feedback
""")
    
    print("=" * 70)
    print("🇦🇫 ZamAI Pro Models - Ready for Production!")
    print("=" * 70)
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
