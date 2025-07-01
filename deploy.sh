#!/bin/bash
# ZamAI Pro Models - Production Deployment Script
# This script sets up and deploys the complete ZamAI ecosystem

set -e  # Exit on any error

echo "🇦🇫 ZamAI Pro Models - Deployment Script"
echo "========================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if required environment variables are set
check_environment() {
    print_status "Checking environment variables..."
    
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        if [ -f "HF-Token.txt" ]; then
            export HUGGINGFACE_TOKEN=$(cat HF-Token.txt)
            print_success "Loaded Hugging Face token from HF-Token.txt"
        else
            print_error "HUGGINGFACE_TOKEN not set and HF-Token.txt not found"
            print_warning "Please set HUGGINGFACE_TOKEN environment variable or create HF-Token.txt"
            exit 1
        fi
    fi
    
    print_success "Environment variables verified"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if command -v python3 &> /dev/null; then
        python3 -m pip install --upgrade pip
        python3 -m pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "Python 3 not found. Please install Python 3.8 or later"
        exit 1
    fi
}

# Validate setup
validate_setup() {
    print_status "Validating setup..."
    python3 scripts/utils/validate_setup.py
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p cache/transformers
    mkdir -p cache/huggingface  
    mkdir -p logs
    mkdir -p ssl
    mkdir -p data/processed
    mkdir -p outputs
    
    print_success "Directories created"
}

# Download and prepare models
prepare_models() {
    print_status "Preparing models..."
    
    # Create model preparation script if it doesn't exist
    cat > prepare_models.py << 'EOF'
#!/usr/bin/env python3
"""Model preparation script for ZamAI deployment"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import login

def download_model(model_id, cache_dir="./cache"):
    """Download and cache a model"""
    print(f"📥 Downloading {model_id}...")
    try:
        # For text generation models
        if "whisper" not in model_id.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                cache_dir=cache_dir,
                torch_dtype="auto",
                device_map="auto"
            )
        else:
            # For Whisper models
            processor = WhisperProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                torch_dtype="auto"
            )
        print(f"✅ Downloaded {model_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_id}: {e}")
        return False

def main():
    # Login to Hugging Face
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token=token)
    
    # Models to download
    models = [
        "openai/whisper-large-v3",
        "mistralai/Mistral-7B-Instruct-v0.3", 
        "microsoft/Phi-3-mini-4k-instruct"
    ]
    
    success_count = 0
    for model_id in models:
        if download_model(model_id):
            success_count += 1
    
    print(f"\n📊 Downloaded {success_count}/{len(models)} models successfully")

if __name__ == "__main__":
    main()
EOF
    
    python3 prepare_models.py
    rm prepare_models.py
}

# Build Docker images (if Docker is available)
build_docker() {
    if command -v docker &> /dev/null; then
        print_status "Building Docker image..."
        docker build -t zamai-pro:latest .
        print_success "Docker image built successfully"
    else
        print_warning "Docker not found. Skipping Docker build"
    fi
}

# Generate SSL certificates for development
generate_ssl_certs() {
    if [ ! -f "ssl/cert.pem" ]; then
        print_status "Generating SSL certificates for development..."
        
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
            -subj "/C=AF/ST=Kabul/L=Kabul/O=ZamAI/OU=Development/CN=zamai.local" 2>/dev/null || {
            print_warning "OpenSSL not available. SSL certificates not generated"
        }
        
        if [ -f "ssl/cert.pem" ]; then
            print_success "SSL certificates generated"
        fi
    fi
}

# Start services
start_services() {
    local deployment_type=${1:-"local"}
    
    case $deployment_type in
        "docker")
            if command -v docker-compose &> /dev/null; then
                print_status "Starting services with Docker Compose..."
                docker-compose up -d
                print_success "Services started with Docker Compose"
                print_status "Services available at:"
                echo "  • Voice Assistant: http://localhost:7860"
                echo "  • Tutor Bot: http://localhost:7861" 
                echo "  • Business Automation: http://localhost:7862"
                echo "  • API Backend: http://localhost:8000"
            else
                print_error "Docker Compose not found"
                exit 1
            fi
            ;;
        "local"|*)
            print_status "Starting local development server..."
            print_status "Use the following commands to start individual services:"
            echo ""
            echo "  # Voice Assistant"
            echo "  python3 zama-hf-pro/voice_assistant/src/app.py"
            echo ""
            echo "  # Tutor Bot"  
            echo "  python3 zama-hf-pro/tutor_bot/src/app.py"
            echo ""
            echo "  # Business Automation"
            echo "  python3 zama-hf-pro/business_automation/src/app.py"
            echo ""
            echo "  # API Backend"
            echo "  python3 zama-hf-pro/fastapi_backend/src/main.py"
            ;;
    esac
}

# Create environment file template
create_env_template() {
    if [ ! -f ".env" ]; then
        print_status "Creating environment template..."
        
        cat > .env << 'EOF'
# ZamAI Pro Models Environment Configuration
# Copy this file to .env and fill in your values

# Hugging Face Configuration
HUGGINGFACE_TOKEN=your_hf_token_here
HF_HOME=./cache/huggingface

# API Configuration  
API_HOST=0.0.0.0
API_PORT=8000

# Gradio Configuration
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

# Model Configuration
MODEL_CACHE_DIR=./cache/transformers
DEFAULT_CHAT_MODEL=mistralai/Mistral-7B-Instruct-v0.3
DEFAULT_WHISPER_MODEL=openai/whisper-large-v3
DEFAULT_PHI_MODEL=microsoft/Phi-3-mini-4k-instruct

# Redis Configuration (for Docker deployment)
REDIS_URL=redis://redis:6379

# Development/Production
ENVIRONMENT=development
DEBUG=true

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs
EOF
        
        print_success "Environment template created (.env)"
        print_warning "Please edit .env with your actual configuration values"
    fi
}

# Main deployment function
main() {
    local deployment_type=${1:-"local"}
    
    print_status "Starting ZamAI Pro Models deployment (${deployment_type} mode)..."
    
    check_environment
    create_directories
    create_env_template
    install_dependencies
    validate_setup
    prepare_models
    generate_ssl_certs
    
    if [ "$deployment_type" == "docker" ]; then
        build_docker
    fi
    
    start_services "$deployment_type"
    
    print_success "🎉 ZamAI Pro Models deployment completed!"
    print_status "Next steps:"
    echo "  1. Edit .env file with your configuration"
    echo "  2. Start the services using the provided commands"
    echo "  3. Visit http://localhost:7860 for the Voice Assistant"
    echo "  4. Check logs/ directory for any issues"
    echo ""
    echo "🇦🇫 Welcome to ZamAI - Afghanistan's Premier AI Platform!"
}

# Parse command line arguments
case "${1:-}" in
    "--help"|"-h")
        echo "Usage: $0 [deployment_type]"
        echo ""
        echo "Deployment types:"
        echo "  local  - Local development setup (default)"
        echo "  docker - Docker-based deployment"
        echo ""
        echo "Examples:"
        echo "  $0              # Local deployment"
        echo "  $0 local        # Local deployment"
        echo "  $0 docker       # Docker deployment"
        exit 0
        ;;
    "docker")
        main "docker"
        ;;
    "local"|"")
        main "local"
        ;;
    *)
        print_error "Unknown deployment type: $1"
        print_status "Use --help for usage information"
        exit 1
        ;;
esac
