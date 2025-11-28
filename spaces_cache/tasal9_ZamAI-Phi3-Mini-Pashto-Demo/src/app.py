import argparse
import json
from pathlib import Path
try:
    import faiss
except ImportError as e:
    raise ImportError(
        "faiss not found. Install with: pip install faiss-cpu (CPU) or pip install faiss-gpu (CUDA)."
    ) from e
import numpy as np
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .config import config

RAG_DEFAULT_K = 4


def load_vector_store(store_dir: str):
    index_path = Path(store_dir) / 'vector.index'
    texts_path = Path(store_dir) / 'texts.json'
    if not index_path.exists() or not texts_path.exists():
        raise FileNotFoundError("Vector store not found. Run ingest first.")
    index = faiss.read_index(str(index_path))
    texts = json.loads(Path(texts_path).read_text(encoding='utf-8'))
    return index, texts


def retrieve(query_emb_model, index, texts, query: str, k: int):
    q_emb = query_emb_model.encode([query], normalize_embeddings=True)
    _, I = index.search(np.array(q_emb, dtype='float32'), k)
    retrieved = [texts[i] for i in I[0]]
    return retrieved


def build_prompt(question: str, contexts):
    ctx_block = "\n\n".join(contexts)
    return f"<|system|>د پاسه ښوونيز ملګری یې. د زده كوونكي پوښتنې ته له ورکړل شویو متونو څخه په Pashto کې واضح، مهربانه او لنډ ځواب ورکړه.<|user|>\nمتنونه:\n{ctx_block}\n\nپوښتنه: {question}\n<|assistant|>"


def create_pipeline(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    return pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=256, do_sample=True, temperature=0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--store', default='rag_store')
    ap.add_argument('--emb_model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer(args.emb_model)
    index, texts = load_vector_store(args.store)
    gen = create_pipeline(args.model)

    def answer(question, k):
        contexts = retrieve(emb_model, index, texts, question, int(k))
        prompt = build_prompt(question, contexts)
        out = gen(prompt)[0]['generated_text'][len(prompt):]
        return "\n".join([out.strip(), "---", "\n\n".join(contexts)])

    iface = gr.Interface(
        fn=answer,
        inputs=[gr.Textbox(label='پوښتنه'), gr.Slider(1,10,value=RAG_DEFAULT_K,step=1,label='Context passages')],
        outputs=gr.Textbox(label='ځواب + سرچينې'),
        title='Pashto Tutor (Phi-3 RAG)',
        description='Fine-tuned Pashto educational assistant with retrieval.'
    )
    iface.launch()

if __name__ == '__main__':
    main()
