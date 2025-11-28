---
base_model: FacebookAI/xlm-roberta-base
library_name: peft
tags:
- base_model:adapter:FacebookAI/xlm-roberta-base
- lora
- transformers
---

# ZamAI XLM-R Pashto LoRA Adapter

This directory contains a lightweight [LoRA](https://arxiv.org/abs/2106.09685) adapter built on top of the
`FacebookAI/xlm-roberta-base` checkpoint. The adapter keeps the full base model frozen and only fine-tunes
two projection matrices (`query`, `value`) per attention layer, making it easy to ship Pashto-specific
updates without redistributing the multi-gigabyte backbone.

> **Status:** the weights are currently untrained (LoRA matrices are initialized to zero so they behave like
> the frozen base). The scaffold is ready for future fine-tuning runs and can be distributed independently
> of the base model.

## Adapter Specs

- **Base model:** `FacebookAI/xlm-roberta-base`
- **Adapter type:** LoRA (via [PEFT](https://github.com/huggingface/peft))
- **Rank (`r`):** 8
- **Alpha:** 16
- **Target modules:** `query`, `value`
- **Dropout:** 0.05
- **Task type:** feature extraction / masked language modeling

Resulting adapter files are only a few megabytes (`adapter_model.safetensors` + `adapter_config.json`).

## Usage

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel

BASE = "FacebookAI/xlm-roberta-base"
ADAPTER = "./adapters/xlm_roberta_pashto_lora"  # or tasal9/ZamAI-Facebook-XLM-Pashto/base_model/... once uploaded

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForMaskedLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

text = "افغانستان يو ___ هېواد دی"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

Swap `ADAPTER` for the Hugging Face path once the adapter is published (e.g.
`tasal9/ZamAI-Facebook-XLM-Pashto/tree/main/adapters/pashto-lora`).

## Fine-tuning Starter

The adapter can be fine-tuned with PEFT + LoRA using any Pashto dataset. Example shell snippet:

```bash
pip install peft datasets

python train_xlm_lora.py \
	--model FacebookAI/xlm-roberta-base \
	--adapter-output adapters/xlm_roberta_pashto_lora \
	--dataset tasal9/ZamAI-Pashto-Dataset-Cleaned
```

Where `train_xlm_lora.py` should:

1. Load the base model + tokenizer
2. Wrap with the same `LoraConfig` described above
3. Train on Pashto text (MLM or downstream task)
4. Save only the adapter weights (`model.save_pretrained(adapter_dir, safe_serialization=True)`)

## Notes & Limitations

- Because the adapter is currently untrained it behaves identically to the frozen base model until you run
	additional fine-tuning. Shipping it now ensures we have a reproducible adapter skeleton and consistent
	hyper-parameters for future runs.
- Loading the adapter never modifies the base weights—removing the adapter reverts to vanilla XLM-R.
- Keep the base checkpoint on Hugging Face (already available under `base_model/`) to avoid storing massive
	binaries in Git.

## Contact

For questions or contributions ping **tasal9** on Hugging Face or open an issue in the ZamAI-Pro-Models repo.
### Framework versions

- PEFT 0.18.0