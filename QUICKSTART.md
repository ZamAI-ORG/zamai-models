# 🚀 Quick Start Guide - Running ZamAI Voice Assistant Locally

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher installed
- pip (Python package manager)
- A HuggingFace account and token

## Step-by-Step Setup

### 1. Get Your HuggingFace Token

1. Visit https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "zamai-local")
4. Select "read" permissions
5. Click "Generate"
6. Copy the token (starts with `hf_`)

### 2. Clone the Repository

```bash
git clone https://github.com/tasal9/ZamAI-Pro-Models.git
cd ZamAI-Pro-Models
```

### 3. Set Up Your Token (Choose ONE method)

**Option A: Environment Variable (Temporary)**
```bash
export HUGGINGFACE_TOKEN="hf_YOUR_TOKEN_HERE"
```

**Option B: Create HF-Token.txt (Recommended)**
```bash
echo "hf_YOUR_TOKEN_HERE" > HF-Token.txt
```

**Option C: Use .env File**
```bash
cp .env.example .env
# Edit .env file and set: HUGGINGFACE_TOKEN=hf_YOUR_TOKEN_HERE
```

### 4. Install Dependencies

**Option A: Using Virtual Environment (Recommended)**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Option B: System-wide Installation**
```bash
pip install -r requirements.txt
```

### 5. Run Pre-flight Check (Optional but Recommended)

```bash
python3 test_startup.py
```

This will verify:
- Python version
- Required modules
- Configuration files
- HuggingFace token
- Port availability

### 6. Start the Application

**Option A: Using the startup script (Easiest)**
```bash
./start_voice_assistant.sh
```

**Option B: Manual start**
```bash
cd zama-hf-pro/voice_assistant/src
python3 app.py
```

**Option C: Using Docker**
```bash
docker-compose up voice-assistant
```

### 7. Access the Application

Once you see this message:
```
Running on local URL:  http://0.0.0.0:7860
```

Open your web browser and navigate to:
- **http://localhost:7860** 
- Or: **http://127.0.0.1:7860**

You should see the ZamAI Voice Assistant interface with three tabs:
- 🎤 Voice Chat - For voice interactions
- 💬 Text Chat - For text-based conversations
- ℹ️ Model Info - Information about loaded models

## 🔧 Troubleshooting

### "localhost refused to connect" Error

This usually means the application isn't running. Check:

1. **Is the application running?**
   ```bash
   # Check if process is running
   ps aux | grep app.py
   
   # Check if port 7860 is in use
   lsof -i :7860
   ```

2. **Check for error messages**
   - Look at the terminal where you started the app
   - Common errors and solutions are in [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

3. **Try a different port**
   ```bash
   export GRADIO_SERVER_PORT=7861
   cd zama-hf-pro/voice_assistant/src
   python3 app.py
   # Then access: http://localhost:7861
   ```

4. **Check your firewall**
   ```bash
   # Ubuntu/Debian
   sudo ufw allow 7860
   ```

### Module Not Found Errors

```bash
# Make sure you installed dependencies
pip install -r requirements.txt

# If using virtual environment, make sure it's activated
source venv/bin/activate
```

### Import Errors for api_client

```bash
# Make sure you're in the correct directory
cd zama-hf-pro/voice_assistant/src
python3 app.py
```

### HuggingFace API Errors

```bash
# Verify your token is set
echo $HUGGINGFACE_TOKEN

# Test the token
curl -H "Authorization: Bearer $HUGGINGFACE_TOKEN" \
  https://huggingface.co/api/whoami-v2
```

## 📊 Verifying Everything Works

After starting the application:

1. **Check the Model Info tab**
   - Click on the "ℹ️ Model Info" tab
   - You should see information about loaded models

2. **Test Text Chat**
   - Go to the "💬 Text Chat" tab
   - Type "Hello" and press Enter
   - Wait for a response (may take a few seconds for first request)

3. **Test Voice Chat** (if you have a microphone)
   - Go to the "🎤 Voice Chat" tab
   - Click "Record" and speak
   - Click "Stop" and then "Process Audio"

## 🎯 Common Startup Issues

| Issue | Solution |
|-------|----------|
| Port already in use | Change port: `export GRADIO_SERVER_PORT=7861` |
| Missing dependencies | Run: `pip install -r requirements.txt` |
| No token configured | Create HF-Token.txt or set HUGGINGFACE_TOKEN |
| Import errors | Run from correct directory: `cd zama-hf-pro/voice_assistant/src` |
| Firewall blocking | Allow port: `sudo ufw allow 7860` |

## 🔍 Still Having Issues?

1. Read the detailed [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide
2. Run the pre-flight check: `python3 test_startup.py`
3. Check the logs for error messages
4. Create an issue on GitHub with:
   - Error messages
   - Steps to reproduce
   - Your system information

## 📚 Next Steps

Once the application is running:
- Explore the different tabs
- Try voice and text interactions
- Read the [README.md](README.md) for more features
- Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for production deployment

## 🌐 Alternative Access Methods

If local setup continues to fail, you can also:
- Use Docker: `docker-compose up`
- Deploy to HuggingFace Spaces
- Use the cloud-hosted version (if available)

---

**Need Help?** 
- 📖 Full documentation: [README.md](README.md)
- 🔧 Troubleshooting: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 🐛 Report issues: [GitHub Issues](https://github.com/tasal9/ZamAI-Pro-Models/issues)
