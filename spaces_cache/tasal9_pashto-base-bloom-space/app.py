import gradio as gr
import time
import threading
import random
from datetime import datetime
from datasets import load_dataset
import pandas as pd
import spaces
import io

# Global state
class TrainingState:
    def __init__(self):
        self.status = "idle"
        self.progress = 0
        self.logs = ["✅ System initialized"]
        self.start_time = None
        self.model_name = "tasal9/pashto-base-bloom"
        self.active_process = None
        self.dataset_loaded = False
        self.dataset_info = "No dataset loaded"
        self.dataset_sample = pd.DataFrame()

    def load_dataset(self):
        try:
            self.logs.append("⏳ Loading dataset: tasal9/ZamAi-Pashto-Datasets-V2")
            dataset = load_dataset("tasal9/ZamAi-Pashto-Datasets-V2")
            self.dataset_loaded = True
            self.dataset_info = f"✅ Dataset loaded!\nName: ZamAi-Pashto-Datasets-V2\nSize: {len(dataset['train'])} examples"
            self.dataset_sample = pd.DataFrame(dataset['train'].select(range(5)))
            self.logs.append(f"📊 {len(dataset['train'])} Pashto examples loaded")
            return True
        except Exception as e:
            self.logs.append(f"❌ Error loading dataset: {str(e)}")
            self.dataset_info = f"Error: {str(e)}"
            return False

    def load_local_file(self, file):
        try:
            ext = file.name.split('.')[-1]
            if ext == "csv":
                df = pd.read_csv(file.name)
            elif ext == "json":
                df = pd.read_json(file.name)
            elif ext == "txt":
                df = pd.DataFrame({"text": open(file.name).read().splitlines()})
            else:
                raise ValueError("Unsupported file format")
            self.dataset_sample = df.head(5)
            self.dataset_info = f"✅ Local file loaded: {file.name}"
            self.dataset_loaded = True
            self.logs.append(f"📁 Local dataset loaded: {file.name}")
            return True
        except Exception as e:
            self.dataset_info = f"❌ Error loading file: {str(e)}"
            self.logs.append(self.dataset_info)
            return False

    def start_training(self, size):
        self.status = "training"
        self.progress = 0
        self.logs = [f"🏋️ Training started at {datetime.now().strftime('%H:%M:%S')}"]
        self.logs.append(f"📝 Data size: {size} characters")
        self.start_time = time.time()

    def start_finetuning(self, size):
        self.status = "fine-tuning"
        self.progress = 0
        self.logs = [f"🎯 Fine-tuning started at {datetime.now().strftime('%H:%M:%S')}"]
        self.logs.append(f"📝 Data size: {size} characters")
        self.start_time = time.time()

    def update_progress(self, progress):
        self.progress = min(100, max(0, progress))
        if progress >= 100:
            self.complete_process()

    def add_log(self, msg):
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        if len(self.logs) > 15:
            self.logs.pop(0)

    def get_status(self):
        return self.status

    @spaces.GPU
    def complete_process(self):
        elapsed = time.time() - self.start_time
        self.add_log(f"🏁 {self.status.capitalize()} completed in {elapsed:.1f}s")
        self.status = "idle"
        self.progress = 100

state = TrainingState()

translations = {
    "English": {
        "title": "🌸 Pashto-Base-Bloom Trainer",
        "load_dataset": "Load Dataset",
        "upload_file": "Upload Local File",
        "status": "Status",
        "preview": "Sample Preview",
        "test_input": "Input",
        "test_output": "Output",
        "test": "Test",
        "train_data": "Training Data",
        "train": "Start Training",
        "finetune_data": "Fine-tuning Data",
        "finetune": "Start Fine-tuning",
        "current_status": "Current Status",
        "progress": "Progress",
        "logs": "Logs",
        "refresh": "🔄 Refresh",
        "export_logs": "📥 Export Logs",
        "language": "Language"
    },
    "پښتو": {
        "title": "🌸 پښتو-بیس-بلوم روزونکی",
        "load_dataset": "ډیټاسټ لوډ کړئ",
        "upload_file": "محلي فایل اپلوډ کړئ",
        "status": "حالت",
        "preview": "نمونه ښودنه",
        "test_input": "ورودی",
        "test_output": "وتی",
        "test": "ازموینه",
        "train_data": "د روزنې معلومات",
        "train": "روزنه پیل کړئ",
        "finetune_data": "د فاین ټیون معلومات",
        "finetune": "فاین ټیون پیل کړئ",
        "current_status": "اوسنی حالت",
        "progress": "پرمختګ",
        "logs": "لاګونه",
        "refresh": "🔄 تازه کړئ",
        "export_logs": "📥 لاګونه ډاونلوډ کړئ",
        "language": "ژبه"
    }
}

def test_model(text):
    if not text.strip():
        return "❗ Enter text to test."
    options = [
        f"Processed: '{text}'",
        f"Model response to: {text}",
        f"Pashto analysis: {len(text)} characters",
        f"✅ Got it: {text}",
        f"Generated: {text}... [simulated]",
        f"🔍 Words: {len(text.split())}"
    ]
    return random.choice(options)

@spaces.GPU
def simulate_process(duration, process_type, data_size):
    if process_type == "train":
        state.start_training(data_size)
    else:
        state.start_finetuning(data_size)
    steps = 10
    for i in range(steps + 1):
        time.sleep(duration / steps)
        state.update_progress(int((i / steps) * 100))
        if i % 3 == 0:
            state.add_log(random.choice([
                f"Batch {i}/{steps}",
                f"Loss: {random.uniform(0.1, 1.0):.3f}",
                f"LR: {random.uniform(1e-5, 1e-3):.6f}",
                f"GPU: {random.randint(60, 95)}% (sim)",
            ]))
    state.complete_process()

def train_model(text):
    if not text.strip():
        return "❌ Add training data.", ""
    if not state.dataset_loaded:
        return "❌ Load dataset first.", ""
    if state.status != "idle":
        return "⏳ Wait for current process.", ""
    threading.Thread(target=simulate_process, args=(15, "train", len(text)), daemon=True).start()
    return "✅ Training started", ""

def finetune_model(text):
    if not text.strip():
        return "❌ Add fine-tuning data.", ""
    if not state.dataset_loaded:
        return "❌ Load dataset first.", ""
    if state.status != "idle":
        return "⏳ Wait for current process.", ""
    threading.Thread(target=simulate_process, args=(10, "fine-tune", len(text)), daemon=True).start()
    return "✅ Fine-tuning started", ""

def load_hf_dataset():
    ok = state.load_dataset()
    return state.dataset_info, state.dataset_sample if ok else pd.DataFrame()

def load_local_dataset(file):
    ok = state.load_local_file(file)
    return state.dataset_info, state.dataset_sample if ok else pd.DataFrame()

def get_current_status():
    return state.get_status(), state.progress / 100, "\n".join(state.logs)

def export_logs():
    df = pd.DataFrame({"Logs": state.logs})
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

with gr.Blocks(title="Pashto Base Bloom Trainer", theme="soft") as demo:
    lang_selector = gr.Dropdown(choices=["English", "پښتو"], value="English", label="Language")
    labels = translations["English"]

    gr.Markdown(f"# {labels['title']}")

    with gr.Tab(labels["load_dataset"]):
        gr.Markdown(f"### {labels['load_dataset']}")
        with gr.Row():
            dataset_btn = gr.Button(labels["load_dataset"])
            dataset_status = gr.Textbox(label=labels["status"], lines=2, interactive=False)
            