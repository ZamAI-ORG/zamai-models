"""Small fill-mask example that prefers a locally-saved model in ./base_model/.

If `./base_model/` exists and contains a saved model, the script loads from
there. Otherwise it will load from Hugging Face (internet required).
"""
from pathlib import Path

from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM


LOCAL_DIR = Path("./base_model")
MODEL_NAME = "FacebookAI/xlm-roberta-base"


def load_pipeline():
    if LOCAL_DIR.exists():
        print(f"Loading model and tokenizer from {LOCAL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
        model = AutoModelForMaskedLM.from_pretrained(LOCAL_DIR)
        return pipeline("fill-mask", model=model, tokenizer=tokenizer)

    print(f"Local model not found, loading {MODEL_NAME} from Hugging Face...")
    return pipeline("fill-mask", model=MODEL_NAME, tokenizer=MODEL_NAME)


def main():
    pipe = load_pipeline()
    mask_token = pipe.tokenizer.mask_token
    text = f"Kabul is the capital of Afghani{mask_token}."
    print("Input:", text)
    results = pipe(text)
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. {r['sequence']} (score={r['score']:.4f})")


if __name__ == "__main__":
    main()
