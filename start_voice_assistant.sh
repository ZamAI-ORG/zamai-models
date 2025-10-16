#!/bin/bash

# ZamAI Voice Assistant Startup Script
# This script helps you start the voice assistant with proper configuration

set -e  # Exit on error

echo "🇦🇫 ZamAI Voice Assistant - Startup Script"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "zama-hf-pro/voice_assistant/src/app.py" ]; then
    echo "❌ Error: Please run this script from the ZamAI-Pro-Models root directory"
    exit 1
fi

# Check for HuggingFace token
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    if [ -f "HF-Token.txt" ]; then
        echo "📝 Loading token from HF-Token.txt..."
        export HUGGINGFACE_TOKEN=$(cat HF-Token.txt | tr -d '[:space:]')
    elif [ -f ".env" ]; then
        echo "📝 Loading configuration from .env file..."
        # Source .env file if it exists
        export $(grep -v '^#' .env | xargs)
    else
        echo "⚠️  Warning: No HUGGINGFACE_TOKEN found!"
        echo "   Please either:"
        echo "   1. Set HUGGINGFACE_TOKEN environment variable"
        echo "   2. Create HF-Token.txt with your token"
        echo "   3. Create .env file from .env.example"
        echo ""
        echo "   Get your token from: https://huggingface.co/settings/tokens"
        echo ""
        read -p "Do you want to continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📥 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "✅ Virtual environment found. Activating..."
    source venv/bin/activate
fi

# Set Gradio configuration
export GRADIO_SERVER_NAME=${GRADIO_SERVER_NAME:-0.0.0.0}
export GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT:-7860}

echo ""
echo "🚀 Starting ZamAI Voice Assistant..."
echo "📍 Server will be available at: http://localhost:$GRADIO_SERVER_PORT"
echo ""
echo "Configuration:"
echo "  - Server: $GRADIO_SERVER_NAME:$GRADIO_SERVER_PORT"
echo "  - Token: ${HUGGINGFACE_TOKEN:0:10}..." 
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Change to the voice assistant directory for proper imports
cd zama-hf-pro/voice_assistant/src

# Start the application
python3 app.py
