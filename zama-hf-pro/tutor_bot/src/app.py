import gradio as gr
from inference import ZamAITutorInference
from prompt_engineering import PashtoEducationPrompts
import os

class ZamAITutorBot:
    def __init__(self):
        self.inference = ZamAITutorInference()
        self.prompts = PashtoEducationPrompts()
        
    def chat_response(self, message, history):
        """Generate educational response"""
        try:
            # Format educational prompt
            educational_prompt = self.prompts.format_educational_prompt(message)
            
            # Generate response
            response = self.inference.generate_response(
                prompt=educational_prompt,
                model="tasal9/ZamAI-Tutor-Bot"
            )
            
            history.append([message, response])
            return "", history
            
        except Exception as e:
            error_msg = f"خطا: {str(e)}"
            history.append([message, error_msg])
            return "", history

# Initialize tutor
tutor = ZamAITutorBot()

# Gradio interface
with gr.Blocks(title="🇦🇫 ZamAI Tutor Bot") as demo:
    gr.Markdown("# 🇦🇫 ZamAI Educational Tutor Bot")
    gr.Markdown("Ask questions in Pashto and get educational responses")
    
    chatbot = gr.Chatbot(
        value=[],
        label="💬 Chat with ZamAI Tutor",
        height=400
    )
    
    msg = gr.Textbox(
        label="📝 Your Question (په پښتو کې پوښتنه وکړئ)",
        placeholder="د ریاضیاتو، ساینس، تاریخ یا ژبې په اړه پوښتنه وکړئ...",
        lines=2
    )
    
    send_btn = gr.Button("📤 Send", variant="primary")
    clear_btn = gr.Button("🗑️ Clear Chat")
    
    send_btn.click(tutor.chat_response, [msg, chatbot], [msg, chatbot])
    msg.submit(tutor.chat_response, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch(share=True)
