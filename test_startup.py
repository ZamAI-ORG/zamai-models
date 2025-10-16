#!/usr/bin/env python3
"""
Quick test script to verify ZamAI Voice Assistant can start
Tests imports and basic configuration without actually starting the server
"""

import sys
import os

print("🔍 ZamAI Voice Assistant - Pre-flight Check")
print("=" * 50)

# Test 1: Python version
print("\n1. Python Version Check...")
python_version = sys.version_info
print(f"   ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version < (3, 8):
    print("   ❌ ERROR: Python 3.8+ required")
    sys.exit(1)

# Test 2: Required modules
print("\n2. Checking Required Modules...")
required_modules = {
    'gradio': 'Gradio UI framework',
    'requests': 'HTTP client',
    'json': 'JSON processing',
    'os': 'OS utilities'
}

missing_modules = []
for module, description in required_modules.items():
    try:
        __import__(module)
        print(f"   ✅ {module:20s} - {description}")
    except ImportError:
        print(f"   ❌ {module:20s} - {description} (MISSING)")
        missing_modules.append(module)

if missing_modules:
    print(f"\n❌ Missing modules: {', '.join(missing_modules)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Check if we can import the app components
print("\n3. Checking Application Components...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'zama-hf-pro/voice_assistant/src'))

try:
    print("   Testing api_client.py...")
    import api_client
    print("   ✅ api_client.py imports successfully")
except Exception as e:
    print(f"   ❌ Error importing api_client: {e}")
    sys.exit(1)

# Test 4: Check for configuration files
print("\n4. Checking Configuration Files...")
config_files = {
    'deployment_config.json': 'Model deployment configuration',
    '.env.example': 'Environment template',
    'requirements.txt': 'Python dependencies'
}

for file, description in config_files.items():
    if os.path.exists(file):
        print(f"   ✅ {file:30s} - {description}")
    else:
        print(f"   ⚠️  {file:30s} - {description} (Not found, may be optional)")

# Test 5: Check for HuggingFace token
print("\n5. Checking HuggingFace Token...")
token = os.getenv('HUGGINGFACE_TOKEN')
if token:
    print(f"   ✅ Token found in environment: {token[:10]}...")
elif os.path.exists('HF-Token.txt'):
    print("   ✅ Token file found: HF-Token.txt")
elif os.path.exists('.env'):
    print("   ⚠️  .env file exists (check if HUGGINGFACE_TOKEN is set)")
else:
    print("   ⚠️  No token configured. Application may have limited functionality.")
    print("      Get token from: https://huggingface.co/settings/tokens")

# Test 6: Check port availability
print("\n6. Checking Port Availability...")
import socket
port = int(os.getenv('GRADIO_SERVER_PORT', 7860))
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('127.0.0.1', port))
sock.close()

if result == 0:
    print(f"   ⚠️  Port {port} is already in use")
    print(f"      Either stop the other application or use a different port:")
    print(f"      export GRADIO_SERVER_PORT=7861")
else:
    print(f"   ✅ Port {port} is available")

# Summary
print("\n" + "=" * 50)
print("✅ Pre-flight check complete!")
print("\nTo start the application:")
print("   ./start_voice_assistant.sh")
print("\nOr manually:")
print("   cd zama-hf-pro/voice_assistant/src")
print("   python3 app.py")
print("\nThen open: http://localhost:7860")
print("=" * 50)
