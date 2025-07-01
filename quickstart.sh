#!/bin/bash
# ZamAI Pro Models - Quick Start Script
# This script provides a quick and easy way to get started with ZamAI Pro Models

set -e  # Exit on any error

echo "🇦🇫 ZamAI Pro Models - Quick Start"
echo "================================="

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "📁 Current directory: $(pwd)"
echo ""

echo "� Available Commands:"
echo "  ./deploy.sh local        # Setup local deployment"
echo "  ./deploy.sh docker       # Setup Docker deployment" 
echo "  python health_check.py   # Check system health"
echo "  python test_deployment_models.py # Test deployed models"
echo ""

echo "🚀 Available Deployment Components:"
echo "  🎤 Voice Assistant: python zama-hf-pro/voice_assistant/src/app.py"
echo "  👨‍🏫 Tutor Bot: python zama-hf-pro/tutor_bot/src/app.py"
echo "  📊 Business Tools: python zama-hf-pro/business_automation/src/app.py"
echo "  🚀 API Server: python zama-hf-pro/fastapi_backend/src/main.py"
echo ""

echo "📋 Deployed Models:"
echo "  💬 Text Generation: mistralai/Mistral-7B-Instruct-v0.3"
echo "  🔉 Speech-to-Text: openai/whisper-large-v3"
echo "  📱 Edge Deployment: microsoft/Phi-3-mini-4k-instruct"
echo ""

echo "✅ Project Status: READY FOR DEPLOYMENT!"
echo "🎯 Next: Run './deploy.sh local' to start local deployment"
echo "📚 For full documentation, see README.md and DEPLOYMENT_GUIDE.md"
