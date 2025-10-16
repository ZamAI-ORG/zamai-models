# 🔧 Troubleshooting Guide - Localhost Connection Issues

This guide helps you resolve "localhost refused to connect" and other common issues with ZamAI Voice Assistant.

## 🚨 Common Issues and Solutions

### Issue 1: "localhost refused to connect"

**Symptoms:**
- Browser shows "This site can't be reached"
- Error: "localhost refused to connect"
- Nothing happens when accessing http://localhost:7860

**Causes:**
1. Application is not running
2. Application crashed during startup
3. Port 7860 is already in use
4. Firewall blocking the connection

**Solutions:**

#### Step 1: Check if the application is running
```bash
# Check if Python app is running on port 7860
netstat -tuln | grep 7860
# OR
lsof -i :7860
```

If nothing is found, the application is not running. Proceed to Step 2.

#### Step 2: Start the application properly

**Option A: Use the startup script (Recommended)**
```bash
# From the project root directory
./start_voice_assistant.sh
```

**Option B: Manual start**
```bash
# Set your HuggingFace token
export HUGGINGFACE_TOKEN="your_token_here"

# Navigate to the voice assistant directory
cd zama-hf-pro/voice_assistant/src

# Run the application
python3 app.py
```

**Option C: Docker deployment**
```bash
# From the project root
docker-compose up voice-assistant
```

#### Step 3: Check for errors in the console

Look for these common errors:

**Error: "ModuleNotFoundError: No module named 'gradio'"**
```bash
# Install dependencies
pip install -r requirements.txt
```

**Error: "No module named 'api_client'"**
```bash
# Make sure you're running from the correct directory
cd zama-hf-pro/voice_assistant/src
python3 app.py
```

**Error: "Invalid token" or authentication errors**
```bash
# Set your HuggingFace token
export HUGGINGFACE_TOKEN="hf_YOUR_TOKEN_HERE"

# Or create HF-Token.txt in the project root with your token
echo "hf_YOUR_TOKEN_HERE" > HF-Token.txt
```

#### Step 4: Check if port 7860 is already in use
```bash
# Linux/Mac
lsof -i :7860

# If port is in use, either:
# 1. Stop the other application
# 2. Use a different port
export GRADIO_SERVER_PORT=7861
python3 app.py
```

#### Step 5: Check firewall settings
```bash
# Ubuntu/Debian
sudo ufw allow 7860

# Check if port is accessible
curl http://localhost:7860
```

---

### Issue 2: Application starts but crashes immediately

**Check the error messages:**

1. **Import errors** - Missing dependencies
   ```bash
   pip install -r requirements.txt
   pip install -r zama-hf-pro/voice_assistant/requirements.txt
   ```

2. **File not found errors** - Wrong directory
   ```bash
   # Always run from the voice_assistant/src directory
   cd zama-hf-pro/voice_assistant/src
   python3 app.py
   ```

3. **Configuration errors** - Missing config file
   ```bash
   # Check if deployment_config.json exists
   ls -la ../../../deployment_config.json
   ```

---

### Issue 3: HuggingFace Token Issues

**Symptoms:**
- API errors
- "Unauthorized" messages
- Models fail to load

**Solutions:**

1. **Get a HuggingFace Token**
   - Visit: https://huggingface.co/settings/tokens
   - Create a new token with "read" permissions
   - Copy the token (starts with "hf_")

2. **Set the token (Choose one method):**

   **Method A: Environment variable**
   ```bash
   export HUGGINGFACE_TOKEN="hf_YOUR_TOKEN_HERE"
   ```

   **Method B: HF-Token.txt file**
   ```bash
   echo "hf_YOUR_TOKEN_HERE" > HF-Token.txt
   ```

   **Method C: .env file**
   ```bash
   cp .env.example .env
   # Edit .env and set: HUGGINGFACE_TOKEN=hf_YOUR_TOKEN_HERE
   ```

3. **Verify token is loaded**
   ```bash
   # Check if token is set
   echo $HUGGINGFACE_TOKEN
   ```

---

### Issue 4: Dependencies Installation Problems

**If pip install fails:**

1. **Update pip**
   ```bash
   pip install --upgrade pip
   ```

2. **Use virtual environment (Recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   venv\Scripts\activate  # Windows
   
   pip install -r requirements.txt
   ```

3. **Install system dependencies (Ubuntu/Debian)**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip
   sudo apt-get install libsndfile1 ffmpeg  # For audio processing
   ```

---

## 🎯 Quick Start Checklist

Before running the application, ensure:

- [ ] Python 3.8+ is installed: `python3 --version`
- [ ] Dependencies are installed: `pip install -r requirements.txt`
- [ ] HuggingFace token is set (one of the methods above)
- [ ] You're in the correct directory: `zama-hf-pro/voice_assistant/src`
- [ ] Port 7860 is available: `lsof -i :7860`

---

## 🚀 Recommended Startup Procedure

1. **From the project root directory:**
   ```bash
   ./start_voice_assistant.sh
   ```

2. **Wait for the message:**
   ```
   Running on local URL:  http://0.0.0.0:7860
   ```

3. **Open your browser:**
   - Navigate to: http://localhost:7860
   - Or: http://127.0.0.1:7860

4. **You should see:**
   - The ZamAI Voice Assistant interface with three tabs:
     - 🎤 Voice Chat
     - 💬 Text Chat
     - ℹ️ Model Info

---

## 📋 Verification Steps

After starting the application:

1. **Check the console output:**
   - No error messages
   - Shows "Running on local URL: http://0.0.0.0:7860"

2. **Test the connection:**
   ```bash
   curl http://localhost:7860
   # Should return HTML content
   ```

3. **Access in browser:**
   - Open http://localhost:7860
   - Interface should load
   - Try the Model Info tab first

---

## 🐛 Debug Mode

For detailed error information:

```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with verbose output
cd zama-hf-pro/voice_assistant/src
python3 -u app.py 2>&1 | tee app.log
```

---

## 📞 Still Having Issues?

1. **Check the logs:**
   ```bash
   cat app.log
   ```

2. **Verify system requirements:**
   - Python 3.8 or higher
   - 4GB+ RAM (8GB+ recommended)
   - Internet connection (for HuggingFace API)

3. **Test with minimal setup:**
   ```bash
   # Test if Gradio works
   python3 -c "import gradio as gr; gr.Interface(lambda x: x, 'text', 'text').launch()"
   ```

4. **Report an issue:**
   - Include error messages
   - Include system information
   - Include steps to reproduce
   - Create an issue at: https://github.com/tasal9/ZamAI-Pro-Models/issues

---

## 🔄 Alternative Deployment Methods

If local deployment continues to fail, try:

### Using Docker:
```bash
docker-compose up voice-assistant
# Access at http://localhost:7860
```

### Using HuggingFace Spaces:
The application is also available at:
- https://huggingface.co/spaces/tasal9/zamai-voice-assistant

---

## 📚 Additional Resources

- [HuggingFace Documentation](https://huggingface.co/docs)
- [Gradio Documentation](https://gradio.app/docs)
- [Project README](README.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
