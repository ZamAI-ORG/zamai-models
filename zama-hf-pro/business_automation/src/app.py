import gradio as gr
from doc_processor import DocumentProcessor
from embedding_tool import EmbeddingTool
from report_generator import ReportGenerator

doc_processor = DocumentProcessor()
embedding_tool = EmbeddingTool()
report_gen = ReportGenerator()

def process_document(file):
    """Process uploaded document"""
    if file is None:
        return "No file uploaded"
    
    # Extract text
    text = doc_processor.extract_text(file.name)
    
    # Generate embeddings
    embeddings = embedding_tool.generate_embeddings(text)
    
    # Create summary report
    report = report_gen.create_summary(text)
    
    return report

with gr.Blocks(title="🇦🇫 ZamAI Business Tools") as demo:
    gr.Markdown("# 🇦🇫 ZamAI Business Document Processor")
    
    with gr.Row():
        file_input = gr.File(label="Upload Document")
        process_btn = gr.Button("Process Document")
    
    output_text = gr.Textbox(label="Analysis Report", lines=10)
    
    process_btn.click(
        process_document,
        inputs=file_input,
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
