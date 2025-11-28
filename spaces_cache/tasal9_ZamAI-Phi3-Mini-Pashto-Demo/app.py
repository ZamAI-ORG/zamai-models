"""
Root-level app.py for Hugging Face Spaces.
This wraps the main application from src/app.py
"""
import os
from pathlib import Path
import spaces

# Import the main app components
from src.app import load_vector_store, retrieve, build_prompt, create_pipeline, RAG_DEFAULT_K

import gradio as gr
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "ZamZeerak-Phi3-Pashto")
STORE_PATH = os.environ.get("STORE_PATH", "rag_store")
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize components
print(f"Loading embedding model: {EMB_MODEL}")
emb_model = SentenceTransformer(EMB_MODEL)

print(f"Loading vector store from: {STORE_PATH}")
try:
    index, texts = load_vector_store(STORE_PATH)
    print(f"Loaded {len(texts)} text chunks")
except FileNotFoundError as e:
    print(f"Warning: {e}")
    print("RAG features will be disabled")
    index, texts = None, []

print(f"Loading model: {MODEL_PATH}")
gen = create_pipeline(MODEL_PATH)
print("Model loaded successfully!")


def answer(question: str, k: int = RAG_DEFAULT_K):
    """Answer a question using RAG."""
    if index is None or not texts:
        return "❌ Vector store not initialized. Please contact the space owner."
    
    try:
        contexts = retrieve(emb_model, index, texts, question, int(k))
        prompt = build_prompt(question, contexts)
        out = gen(prompt)[0]['generated_text'][len(prompt):]
        
        # Format response with sources
        response = f"{out.strip()}\n\n---\n\n**د سرچينې (Sources):**\n\n"
        for i, ctx in enumerate(contexts, 1):
            response += f"{i}. {ctx[:200]}...\n\n"
        
        return response
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Pashto Tutor (Phi-3 RAG)", theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>📚 Pashto Educational Tutor</h1>
        <p>د پښتو ښوونيز ملګری - Powered by Phi-3 with RAG</p>
        <p><em>Fine-tuned on Pashto educational content</em></p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="پوښتنه (Question)",
                placeholder="خپله پوښتنه دلته وليکئ... (Enter your question here...)",
                lines=3
            )
            k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=RAG_DEFAULT_K,
                step=1,
                label="Number of Context Passages"
            )
            submit_btn = gr.Button("ځواب ترلاسه کړئ (Get Answer)", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="ځواب + سرچينې (Answer + Sources)",
                lines=15,
                interactive=False
            )
    
    # Examples
    gr.Examples(
        examples=[
            ["د ریاضی په اړه راته ووایه", 4],
            ["د پښتو ګرامر څه دی؟", 4],
            ["د علوم اساسی مفاهیم تشریح کړه", 4],
        ],
        inputs=[question_input, k_slider]
    )
    
    # Event handlers
    submit_btn.click(
        fn=answer,
        inputs=[question_input, k_slider],
        outputs=output
    )
    
    question_input.submit(
        fn=answer,
        inputs=[question_input, k_slider],
        outputs=output
    )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #ddd;">
        <p>⚠️ This is an educational tool. Verify information with official sources.</p>
        <p>🔍 Uses retrieval-augmented generation (RAG) for accurate, context-aware responses.</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
