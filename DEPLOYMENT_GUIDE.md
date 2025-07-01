# 🚀 ZamAI Pro Models - Production Deployment Guide

## Overview

This guide will help you deploy ZamAI Pro Models in a production environment with high availability, security, and performance.

## 🆕 New Models Available

### Speech Recognition
- **tasal9/ZamAI-Whisper-v3-Pashto**
  - Based on: openai/whisper-large-v3
  - Purpose: Pashto speech recognition
  - Push script: `scripts/utils/push_whisper_pashto.py`
  - Test script: `scripts/testing/test_whisper_pashto.py`

### Text Generation
- **tasal9/ZamAI-Phi-3-Mini-Pashto**
  - Based on: microsoft/Phi-3-mini-4k-instruct
  - Purpose: Lightweight Pashto instruction model
  - Push script: `scripts/utils/push_phi3_mini_pashto.py`
  - Test script: `scripts/testing/test_phi3_mini_pashto.py`

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ or CentOS 8+
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 100GB+ SSD storage
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Network**: Stable internet connection

### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.8+
- nginx 1.18+
- Git

## 🔧 Quick Production Setup

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install nginx
sudo apt install nginx -y

# Install other dependencies
sudo apt install git curl wget htop -y
```

### 2. Deploy ZamAI

```bash
# Clone repository
git clone https://github.com/tasal9/ZamAI-Pro-Models.git
cd ZamAI-Pro-Models

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Set Hugging Face token
echo "your_hf_token_here" > HF-Token.txt

# Deploy with Docker
./deploy.sh docker

# Verify deployment
./health_check.py
```

### 3. Configure nginx (Production)

```bash
# Copy nginx configuration
sudo cp nginx.conf /etc/nginx/sites-available/zamai
sudo ln -s /etc/nginx/sites-available/zamai /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Generate SSL certificates (Let's Encrypt)
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com

# Test nginx configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

## 🔐 Security Configuration

### 1. Firewall Setup

```bash
# Enable UFW
sudo ufw enable

# Allow necessary ports
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS

# Block direct access to application ports
sudo ufw deny 7860
sudo ufw deny 7861
sudo ufw deny 7862
sudo ufw deny 8000
```

### 2. SSL/TLS Configuration

Update your `.env` file:
```bash
SSL_ENABLED=true
SSL_CERT_FILE=/etc/letsencrypt/live/your-domain.com/fullchain.pem
SSL_KEY_FILE=/etc/letsencrypt/live/your-domain.com/privkey.pem
```

### 3. API Security

```bash
# Generate secure keys
export SECRET_KEY=$(openssl rand -hex 32)
export JWT_SECRET=$(openssl rand -hex 32)

# Update .env file
echo "SECRET_KEY=$SECRET_KEY" >> .env
echo "JWT_SECRET=$JWT_SECRET" >> .env
```

## 📊 Monitoring Setup

### 1. System Monitoring

```bash
# Install monitoring tools
sudo apt install htop iotop nethogs -y

# Setup log rotation
sudo tee /etc/logrotate.d/zamai << EOF
/home/ubuntu/ZamAI-Pro-Models/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 ubuntu ubuntu
}
EOF
```

### 2. Health Monitoring

```bash
# Add health check to crontab
crontab -e
# Add this line:
# */5 * * * * cd /home/ubuntu/ZamAI-Pro-Models && python3 health_check.py >> logs/health.log 2>&1
```

### 3. Application Monitoring

Enable monitoring in `.env`:
```bash
ENABLE_METRICS=true
WANDB_PROJECT=zamai-production
WANDB_ENTITY=your-wandb-entity
```

## 🔄 Backup Strategy

### 1. Data Backup

```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backups/zamai-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup configurations
cp -r ZamAI-Pro-Models/configs $BACKUP_DIR/
cp ZamAI-Pro-Models/.env $BACKUP_DIR/
cp ZamAI-Pro-Models/HF-Token.txt $BACKUP_DIR/

# Backup logs
cp -r ZamAI-Pro-Models/logs $BACKUP_DIR/

# Backup database (if using)
if [ -f "zamai.db" ]; then
    cp zamai.db $BACKUP_DIR/
fi

# Create archive
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

### 2. Automated Backups

```bash
# Add to crontab
# 0 2 * * * /home/ubuntu/backup.sh
```

## 🚀 Performance Optimization

### 1. GPU Optimization

If using GPU:
```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-470 -y

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

Update `docker-compose.yml` to use GPU:
```yaml
services:
  voice-assistant:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. Model Caching

```bash
# Pre-download models
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperProcessor, WhisperForConditionalGeneration

models = [
    'openai/whisper-large-v3',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'microsoft/Phi-3-mini-4k-instruct'
]

for model_id in models:
    print(f'Downloading {model_id}...')
    if 'whisper' in model_id:
        WhisperProcessor.from_pretrained(model_id)
        WhisperForConditionalGeneration.from_pretrained(model_id)
    else:
        AutoTokenizer.from_pretrained(model_id)
        AutoModelForCausalLM.from_pretrained(model_id)
"
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Check memory usage
   free -h
   
   # Restart services
   docker-compose restart
   ```

2. **Model Loading Errors**
   ```bash
   # Check HF token
   python3 -c "from huggingface_hub import whoami; print(whoami())"
   
   # Clear cache
   rm -rf cache/
   ```

3. **Service Not Responding**
   ```bash
   # Check service logs
   docker-compose logs voice-assistant
   
   # Restart specific service
   docker-compose restart voice-assistant
   ```

### Log Locations

- **Application Logs**: `logs/`
- **Docker Logs**: `docker-compose logs [service]`
- **nginx Logs**: `/var/log/nginx/`
- **System Logs**: `/var/log/syslog`

## 📈 Scaling

### Horizontal Scaling

1. **Load Balancer Setup**
   ```nginx
   upstream zamai_backend {
       server server1.example.com:8000;
       server server2.example.com:8000;
       server server3.example.com:8000;
   }
   ```

2. **Database Clustering**
   - Use PostgreSQL with read replicas
   - Redis Cluster for caching

### Vertical Scaling

1. **Resource Allocation**
   ```yaml
   # docker-compose.yml
   services:
     voice-assistant:
       deploy:
         resources:
           limits:
             memory: 8G
             cpus: '4'
   ```

## 🛠️ Maintenance

### Regular Tasks

1. **Weekly**
   ```bash
   # Update system packages
   sudo apt update && sudo apt upgrade -y
   
   # Clean Docker
   docker system prune -f
   
   # Rotate logs
   sudo logrotate -f /etc/logrotate.d/zamai
   ```

2. **Monthly**
   ```bash
   # Update Docker images
   docker-compose pull
   docker-compose up -d
   
   # Check disk usage
   df -h
   
   # Review security updates
   sudo apt list --upgradable
   ```

## 📞 Support

For production support:
- **Documentation**: Check `README.md` and model cards

## 💻 Model Usage Examples

### Whisper Pashto Speech Recognition
```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

# Load model and processor
processor = AutoProcessor.from_pretrained("tasal9/ZamAI-Whisper-v3-Pashto")
model = AutoModelForSpeechSeq2Seq.from_pretrained("tasal9/ZamAI-Whisper-v3-Pashto")

# For GPU usage
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Transcribe Pashto audio
import librosa
audio_data, sampling_rate = librosa.load("path/to/pashto_audio.wav", sr=16000)
input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)

# Generate transcription
with torch.no_grad():
    generated_ids = model.generate(input_features=input_features, language="ps", task="transcribe")

# Decode the generated ids
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Transcription: {transcription}")
```

### Phi-3 Mini Pashto Text Generation
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("tasal9/ZamAI-Phi-3-Mini-Pashto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("tasal9/ZamAI-Phi-3-Mini-Pashto", trust_remote_code=True)

# Generate text in Pashto
prompt = "په پښتو کې ماته وایاست چې افغانستان څه ډول هیواد دی؟"  # "Tell me in Pashto what kind of country Afghanistan is?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Response: {response}")
```
- **Health Check**: Run `./health_check.py`
- **Logs**: Check `logs/` directory
- **Community**: Create GitHub issue

---

**Production Checklist**
- [ ] SSL certificates configured
- [ ] Firewall rules applied
- [ ] Monitoring enabled
- [ ] Backup strategy implemented
- [ ] Health checks running
- [ ] Performance optimized
- [ ] Security hardened
- [ ] Documentation updated
