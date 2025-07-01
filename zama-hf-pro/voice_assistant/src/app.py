import gradio as gr
from api_client import HFInferenceClient
import os
import json

class ZamAIVoiceAssistant:
    def __init__(self):
        token = os.getenv("HUGGINGFACE_TOKEN", "")
        self.hf_client = HFInferenceClient(token=token)
        
        # Load deployment config
        self.load_config()
        
    def load_config(self):
        """Load deployment configuration"""
        try:
            with open("../../../deployment_config.json", "r") as f:
                config = json.load(f)
                self.models = config["deployment_models"]
                self.pipeline_config = config["voice_assistant_pipeline"]
        except Exception as e:
            print(f"Warning: Could not load config, using defaults: {e}")
            # Default models
            self.models = {
                "speech_to_text": {"primary": "openai/whisper-large-v3"},
                "text_generation": {"primary": "mistralai/Mistral-7B-Instruct-v0.3"}
            }
        
    def process_audio(self, audio):
        """Complete STT → Understanding → Response → TTS workflow"""
        try:
            if audio is None:
                return "Please record some audio first.", None
            
            # Step 1: Speech to Text using Whisper Large v3
            stt_model = self.models["speech_to_text"]["primary"]
            text = self.hf_client.stt_process(audio, model=stt_model)
            
            if not text:
                return "Could not understand the audio. Please try speaking more clearly.", None
            
            # Step 2: Language Understanding & Response Generation
            llm_model = self.models["text_generation"]["primary"]
            
            # Create context-aware prompt for Pashto conversation
            prompt = f"""You are ZamAI, an AI assistant for Afghanistan that speaks Pashto. 
Respond naturally and helpfully to the user's question.

User: {text}
ZamAI:"""
            
            response = self.hf_client.generate_text(
                prompt=prompt,
                model=llm_model,
                max_length=200,
                temperature=0.7
            )
            
            # Clean up response
            if "ZamAI:" in response:
                response = response.split("ZamAI:")[-1].strip()
            
            # Step 3: Text to Speech (using available TTS model)
            try:
                audio_out = self.hf_client.tts_process(response)
            except Exception as tts_error:
                print(f"TTS Error: {tts_error}")
                audio_out = None
            
            return response, audio_out
            
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            print(error_msg)
            return error_msg, None
    
    def chat_text(self, message, history):
        """Text-based chat interface"""
        try:
            llm_model = self.models["text_generation"]["primary"]
            
            # Build conversation context
            context = ""
            for user_msg, bot_msg in history:
                context += f"User: {user_msg}\nZamAI: {bot_msg}\n"
            
            prompt = f"""You are ZamAI, an AI assistant for Afghanistan. Respond helpfully in the user's language.

{context}User: {message}
ZamAI:"""
            
            response = self.hf_client.generate_text(
                prompt=prompt,
                model=llm_model,
                max_length=300,
                temperature=0.7
            )
            
            # Clean up response
            if "ZamAI:" in response:
                response = response.split("ZamAI:")[-1].strip()
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = f"""
🤖 **Current Models:**
- **Speech Recognition**: {self.models.get('speech_to_text', {}).get('primary', 'N/A')}
- **Text Generation**: {self.models.get('text_generation', {}).get('primary', 'N/A')}
- **Features**: Speech-to-text + Understanding + Response Generation + Text-to-speech

📊 **Capabilities:**
- Pashto speech recognition
- Multilingual understanding
- Context-aware responses
- Natural conversation flow
        """
        return info.strip()

# Initialize assistant
assistant = ZamAIVoiceAssistant()

# Create Gradio interface with multiple tabs
with gr.Blocks(title="🇦🇫 ZamAI Pro Voice Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇦🇫 ZamAI Pro Voice Assistant")
    gr.Markdown("*Advanced AI assistant with speech recognition, understanding, and response generation*")
    
    with gr.Tab("🎤 Voice Chat"):
        gr.Markdown("### Speak naturally and get AI-powered responses")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone"], 
                    type="filepath", 
                    label="🎤 Record your voice"
                )
                process_btn = gr.Button("🔄 Process Audio", variant="primary")
                
            with gr.Column():
                text_output = gr.Textbox(
                    label="📝 AI Response", 
                    lines=4,
                    placeholder="AI response will appear here..."
                )
                audio_output = gr.Audio(label="🔊 Voice Response")
        
        process_btn.click(
            fn=assistant.process_audio,
            inputs=[audio_input],
            outputs=[text_output, audio_output]
        )
    
    with gr.Tab("💬 Text Chat"):
        gr.Markdown("### Chat directly with text")
        
        chatbot = gr.Chatbot(label="ZamAI Conversation", height=400)
        msg = gr.Textbox(
            label="Your message",
            placeholder="Type your message here...",
            lines=2
        )
        
        msg.submit(assistant.chat_text, [msg, chatbot], [chatbot])
        msg.submit(lambda: "", None, [msg])
    
    with gr.Tab("ℹ️ Model Info"):
        gr.Markdown("### Current Model Configuration")
        model_info = gr.Textbox(
            value=assistant.get_model_info(),
            label="Model Information",
            lines=10,
            interactive=False
        )
        refresh_btn = gr.Button("🔄 Refresh Info")
        refresh_btn.click(
            fn=assistant.get_model_info,
            outputs=[model_info]
        )

# Health check endpoint
@demo.api()
def health():
    return {"status": "healthy", "models_loaded": True}

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
