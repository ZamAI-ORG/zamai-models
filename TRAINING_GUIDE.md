
# ZamAI Model Training Guide

## Models That Need Training

Based on your configurations, these models need to be fine-tuned:

### 1. Pashto Chat Model
- **Model ID**: tasal9/zamai-pashto-chat-8b
- **Base Model**: meta-llama/Llama-3.1-8B-Instruct
- **Purpose**: Pashto language conversational AI
- **Training Script**: fine-tuning/train_pashto_chat.py

### Training Steps:

1. **Prepare Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set HF Token**:
   ```bash
   # Add your token to HF-Token.txt
   echo "your_token_here" > HF-Token.txt
   ```

3. **Check Dataset Access**:
   ```bash
   python -c "from datasets import load_dataset; print(load_dataset('tasal9/ZamAI_Pashto_Dataset'))"
   ```

4. **Start Training**:
   ```bash
   python fine-tuning/train_pashto_chat.py
   ```

5. **Monitor Progress**:
   - Check wandb dashboard
   - Monitor GPU usage
   - Watch loss curves

### Training Configuration:
- **LoRA Rank**: 64
- **Learning Rate**: 2e-5
- **Batch Size**: 2 (with gradient accumulation)
- **Epochs**: 3
- **Expected Time**: 2-4 hours on A100

### After Training:
1. Model will be automatically pushed to HF Hub
2. Test the model using inference scripts
3. Deploy to production endpoints

## Troubleshooting:

### Common Issues:
1. **CUDA out of memory**: Reduce batch_size in config
2. **Dataset not found**: Check HF token permissions
3. **Push to hub fails**: Verify token has write access

### Support:
- Check logs in outputs/ directory
- Review wandb runs for debugging
- Test locally before deploying
