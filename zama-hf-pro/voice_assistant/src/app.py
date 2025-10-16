import gradio as gr
from api_client import HFInferenceClient
import os
import json
from datetime import datetime
import time

class ZamAIVoiceAssistant:
    def __init__(self):
        token = os.getenv("HUGGINGFACE_TOKEN", "")
        self.hf_client = HFInferenceClient(token=token)
        self.conversation_count = 0
        self.start_time = datetime.now()
        
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
            
            # Gradio returns audio as tuple (file_path, sample_rate) or just file_path
            audio_file = audio if isinstance(audio, str) else audio[0] if isinstance(audio, tuple) else None
            
            if not audio_file:
                return "Invalid audio format.", None
            
            # Step 1: Speech to Text using Whisper Large v3
            stt_model = self.models["speech_to_text"]["primary"]
            text = self.hf_client.stt_process(audio_file, model=stt_model)
            
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
    
    def chat_text(self, message, history, temperature, max_tokens):
        """Text-based chat interface with customizable parameters"""
        try:
            if not message or not message.strip():
                return history, self.get_stats()
                
            llm_model = self.models["text_generation"]["primary"]
            
            # Build conversation context from messages format
            context = ""
            for msg in history:
                if msg.get("role") == "user":
                    context += f"User: {msg.get('content', '')}\n"
                elif msg.get("role") == "assistant":
                    context += f"ZamAI: {msg.get('content', '')}\n"
            
            prompt = f"""You are ZamAI, an AI assistant for Afghanistan. Respond helpfully in the user's language.

{context}User: {message}
ZamAI:"""
            
            start_time = time.time()
            response = self.hf_client.generate_text(
                prompt=prompt,
                model=llm_model,
                max_length=int(max_tokens),
                temperature=float(temperature)
            )
            response_time = time.time() - start_time
            
            # Clean up response
            if "ZamAI:" in response:
                response = response.split("ZamAI:")[-1].strip()
            
            # Add response time info
            response += f"\n\n_⏱️ Response time: {response_time:.2f}s_"
            
            # Return updated history in messages format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            self.conversation_count += 1
            
            return history, self.get_stats()
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, self.get_stats()
    
    def clear_chat(self):
        """Clear chat history"""
        return [], self.get_stats()
    
    def get_stats(self):
        """Get conversation statistics"""
        uptime = datetime.now() - self.start_time
        return f"""📊 **Session Stats:**
- Conversations: {self.conversation_count}
- Uptime: {str(uptime).split('.')[0]}
- Model: {self.models.get('text_generation', {}).get('primary', 'N/A')}"""
    
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
        
        chatbot = gr.Chatbot(label="ZamAI Conversation", height=400, type='messages')
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

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
