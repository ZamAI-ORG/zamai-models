"""
Root-level app.py for Hugging Face Spaces.
This wraps the main application from src/app.py
"""
import os

# Import the main app components
from src.app import load_vector_store, retrieve, build_prompt, create_pipeline, RAG_DEFAULT_K

import gradio as gr

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "tasal9/ZamAI-Phi-3-Mini-Pashto")
STORE_PATH = os.environ.get("STORE_PATH", "rag_store")
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_CONTEXTS = [
    "د پښتو ژبې ګرامر کې نوم، فعل او صفت د جملې د جوړښت بنسټ جوړوي. زده کوونکي باید هره ورځ لږ تر لږه یوه پاراګراف تحریر کړي تر څو لیکنه یې روانه شي.",
    "د ریاضي بنسټیز مفاهیم لکه جمع، تفریق، ضرب او تقسیم د لومړنیو ټولګیو لپاره مهم دي. عملي تمرین او د ورځني ژوند مثالونه زده کړه اسانه کوي.",
    "د طبعي علومو درسونه باید د چاپیریال ساتنې، اوبو د اهمیت، او د نباتاتو د ودې د پړاوونو تشریح ولري تر څو ماشومان له خپل چاپیریال سره مینه پیدا کړي.",
    "Islamic studies lessons highlight honesty, respect for parents, and community service. Encourage students to connect each حدیث with a real-life example.",
    "Pashto literature classes can include poems from Khushal Khan Khattak and Rahman Baba. Discussing د شاعر پیغام learners ته د اخلاقو، میړانې او ورورولۍ ارزښتونه یادوي.",
]

# Initialize components
print(f"Loading embedding model: {EMB_MODEL}")
if SentenceTransformer:
    emb_model = SentenceTransformer(EMB_MODEL)
else:
    emb_model = None
    print("Warning: sentence_transformers not installed. RAG features disabled.")

print(f"Loading vector store from: {STORE_PATH}")
try:
    index, texts = load_vector_store(STORE_PATH, emb_model=emb_model, fallback_texts=DEFAULT_CONTEXTS)
    print(f"Loaded {len(texts)} text chunks")
except (FileNotFoundError, ImportError) as e:
    print(f"Warning: {e}")
    print("RAG features will be disabled")
    index, texts = None, []

print(f"Loading model: {MODEL_PATH}")
gen = create_pipeline(MODEL_PATH)
print("Model loaded successfully!")


def answer(question: str, k: int = RAG_DEFAULT_K):
    """Answer a question using RAG."""
    if index is None or not texts or emb_model is None:
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
demo = gr.Blocks(theme=gr.themes.Soft())
with demo:
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
