# 🔴 LOCALHOST REFUSED TO CONNECT - Quick Fix Guide

> **You're seeing:** "This site can't be reached" or "localhost refused to connect"  
> **You want:** To access the ZamAI Voice Assistant at http://localhost:7860

## 🎯 Quick Fix (Most Common)

The application isn't running. Follow these steps:

### Step 1: Get Your HuggingFace Token (If you don't have one)
```bash
# Visit: https://huggingface.co/settings/tokens
# Create a token with "read" permission
# Copy it (starts with hf_)
```

### Step 2: Set Your Token
```bash
# Save to HF-Token.txt (easiest)
echo "hf_YOUR_TOKEN_HERE" > HF-Token.txt

# OR set as environment variable
export HUGGINGFACE_TOKEN="hf_YOUR_TOKEN_HERE"
```

### Step 3: Install Dependencies
```bash
# Install required packages
pip install gradio requests

# Or install everything
pip install -r requirements.txt
```

### Step 4: Start the Application
```bash
# Use the startup script (recommended)
./start_voice_assistant.sh

# OR manually
cd zama-hf-pro/voice_assistant/src
python3 app.py
```

### Step 5: Wait for Success Message
You should see:
```
* Running on local URL:  http://0.0.0.0:7860
```

### Step 6: Open Browser
Go to: **http://localhost:7860**

---

## 🔍 Still Not Working?

### Check if Application is Running
```bash
# Is it running?
ps aux | grep app.py

# Is port 7860 in use?
lsof -i :7860
```

**If nothing appears:** The application is not running. Check for errors in Step 4.

### Common Errors and Fixes

#### Error: "ModuleNotFoundError: No module named 'gradio'"
```bash
pip install gradio requests
```

#### Error: "cannot import name 'HFInferenceClient'"
```bash
# Wrong directory - navigate to the right place
cd zama-hf-pro/voice_assistant/src
python3 app.py
```

#### Error: "Address already in use"
```bash
# Port 7860 is busy - use different port
export GRADIO_SERVER_PORT=7861
python3 app.py
# Then open: http://localhost:7861
```

#### Error: Application starts but crashes
```bash
# Run pre-flight check to diagnose
python3 test_startup.py
```

---

## 📋 Verification Checklist

Before reporting issues, verify:

- [ ] Python 3.8+ installed: `python3 --version`
- [ ] Dependencies installed: `pip list | grep gradio`
- [ ] Token is set: `echo $HUGGINGFACE_TOKEN` or check HF-Token.txt exists
- [ ] In correct directory: `pwd` should show the project root
- [ ] Port 7860 is free: `lsof -i :7860` shows nothing
- [ ] Application runs without crashing
- [ ] You see "Running on local URL" message

---

## 🚀 Full Setup Guide

For complete setup instructions:
1. **Quick Setup:** [QUICKSTART.md](QUICKSTART.md)
2. **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. **Success Indicators:** [SUCCESS_INDICATORS.md](SUCCESS_INDICATORS.md)
4. **Full Documentation:** [README.md](README.md)

---

## 🆘 Emergency Quick Test

Just want to verify it can work?

```bash
# Install minimal dependencies
pip install gradio requests

# Create a dummy token file
echo "hf_PLACEHOLDER" > HF-Token.txt

# Run test
python3 test_startup.py

# If all checks pass except token warning, you're ready!
# Just add your real token and run:
cd zama-hf-pro/voice_assistant/src
python3 app.py
```

---

## 📞 Need More Help?

1. Run diagnostics: `python3 test_startup.py`
2. Read: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. Report issue with error messages: [GitHub Issues](https://github.com/tasal9/ZamAI-Pro-Models/issues)

---

**TL;DR:** Application must be running to access localhost. Run `./start_voice_assistant.sh` first!
