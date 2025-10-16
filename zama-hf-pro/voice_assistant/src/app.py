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

# Create Enhanced Gradio interface
custom_css = """
.header-text {text-align: center; color: #2563eb;}
.stat-box {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white;}
"""

with gr.Blocks(title="🇦🇫 ZamAI Pro Voice Assistant", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    # 🇦🇫 ZamAI Pro Voice Assistant
    ### *د افغانستان AI مرستیال - Afghanistan's Advanced AI Assistant*
    
    **Powered by:** Whisper Large v3 • Mistral 7B • Hugging Face Inference
    """, elem_classes="header-text")
    
    with gr.Tab("💬 Smart Chat"):
        gr.Markdown("### 🤖 Intelligent Conversation with Advanced Controls")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="ZamAI Conversation", 
                    height=500, 
                    type='messages',
                    avatar_images=(None, "🤖"),
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Ask me anything in English, Pashto, or Dari...",
                        lines=2,
                        scale=4
                    )
                    with gr.Column(scale=1, min_width=100):
                        send_btn = gr.Button("� Send", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ Clear", size="lg")
            
            with gr.Column(scale=1):
                stats_box = gr.Markdown(assistant.get_stats(), elem_classes="stat-box")
                
                gr.Markdown("### ⚙️ Generation Settings")
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="🌡️ Temperature (Creativity)",
                    info="Higher = more creative, Lower = more focused"
                )
                
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=300,
                    step=50,
                    label="📏 Max Tokens (Length)",
                    info="Maximum response length"
                )
                
                gr.Markdown("### 💡 Quick Prompts")
                with gr.Column():
                    example_btn1 = gr.Button("👋 Introduce yourself", size="sm")
                    example_btn2 = gr.Button("📚 Explain AI", size="sm")
                    example_btn3 = gr.Button("🇦🇫 Tell me about Afghanistan", size="sm")
                    example_btn4 = gr.Button("� Help with coding", size="sm")
        
        # Event handlers for chat
        send_btn.click(assistant.chat_text, [msg, chatbot, temperature, max_tokens], [chatbot, stats_box])
        send_btn.click(lambda: "", None, [msg])
        msg.submit(assistant.chat_text, [msg, chatbot, temperature, max_tokens], [chatbot, stats_box])
        msg.submit(lambda: "", None, [msg])
        clear_btn.click(assistant.clear_chat, None, [chatbot, stats_box])
        
        # Quick prompt handlers
        example_btn1.click(lambda: "Hello! Please introduce yourself and tell me what you can do.", None, msg)
        example_btn2.click(lambda: "Can you explain what artificial intelligence is in simple terms?", None, msg)
        example_btn3.click(lambda: "Tell me about Afghanistan's culture and history.", None, msg)
        example_btn4.click(lambda: "I need help writing a Python function. Can you assist?", None, msg)
    
    with gr.Tab("🎤 Voice Chat"):
        gr.Markdown("### 🗣️ Speak naturally and get AI-powered responses")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone"], 
                    type="filepath", 
                    label="🎤 Record your voice",
                    show_download_button=True
                )
                process_btn = gr.Button("🔄 Process Audio", variant="primary", size="lg")
                gr.Markdown("**Tip:** Speak clearly and wait for the model to process your audio.")
                
            with gr.Column():
                text_output = gr.Textbox(
                    label="📝 Transcribed Text & AI Response", 
                    lines=8,
                    placeholder="Your speech will be transcribed and AI will respond here...",
                    show_copy_button=True
                )
                audio_output = gr.Audio(label="🔊 Voice Response (if available)")
        
        process_btn.click(
            fn=assistant.process_audio,
            inputs=[audio_input],
            outputs=[text_output, audio_output]
        )
    
    with gr.Tab("ℹ️ Model Info"):
        gr.Markdown("### 🤖 System Information & Model Configuration")
        
        with gr.Row():
            with gr.Column():
                model_info = gr.Textbox(
                    value=assistant.get_model_info(),
                    label="📊 Current Models",
                    lines=12,
                    interactive=False
                )
                refresh_btn = gr.Button("🔄 Refresh Info", variant="secondary")
                refresh_btn.click(
                    fn=assistant.get_model_info,
                    outputs=[model_info]
                )
            
            with gr.Column():
                gr.Markdown("""
                ### 🌟 Features
                
                ✅ **Multi-language Support**
                - Pashto (پښتو)
                - Dari (دری)
                - English
                - And more!
                
                ✅ **Advanced Capabilities**
                - Speech-to-Text (Whisper Large v3)
                - Natural Language Understanding
                - Context-aware responses
                - Text-to-Speech synthesis
                
                ✅ **Customization**
                - Adjustable temperature
                - Configurable response length
                - Real-time statistics
                
                ### 📚 Use Cases
                - Language learning
                - Document assistance
                - Code help
                - General Q&A
                - Cultural information
                
                ### 🔗 Links
                - [GitHub Repository](https://github.com/tasal9/ZamAI-Pro-Models)
                - [Hugging Face](https://huggingface.co/tasal9)
                - [Report Issues](https://github.com/tasal9/ZamAI-Pro-Models/issues)
                """)
    
    with gr.Tab("📖 About"):
        gr.Markdown("""
        # About ZamAI Pro Voice Assistant
        
        ## 🎯 Mission
        Building Afghanistan's premier AI ecosystem to empower education, business, and daily life through cutting-edge AI technology.
        
        ## 🏗️ Architecture
        ```
        User Input (Voice/Text)
            ↓
        Speech Recognition (Whisper Large v3)
            ↓
        Natural Language Understanding
            ↓
        Response Generation (Mistral 7B / Phi-3)
            ↓
        Text-to-Speech Synthesis
            ↓
        Output (Voice/Text)
        ```
        
        ## 🤖 Models Used
        
        | Model | Purpose | Size | Provider |
        |-------|---------|------|----------|
        | Whisper Large v3 | Speech Recognition | 1.5B | OpenAI |
        | Mistral 7B Instruct | Text Generation | 7B | Mistral AI |
        | Phi-3 Mini | Lightweight AI | 3.8B | Microsoft |
        | SpeechT5 | Text-to-Speech | 200M | Microsoft |
        
        ## 🙏 Acknowledgments
        - **Hugging Face** - AI infrastructure and models
        - **OpenAI** - Whisper model
        - **Mistral AI** - Language models
        - **Microsoft** - Phi-3 and SpeechT5
        - **Afghan AI Community** - Cultural guidance and support
        
        ## 📞 Contact
        - **Developer:** Yaqoob Tasal (@tasal9)
        - **Email:** tasal9@huggingface.co
        - **GitHub:** [ZamAI-Pro-Models](https://github.com/tasal9/ZamAI-Pro-Models)
        
        ## 📄 License
        Apache 2.0 License - Open source and free to use
        
        ---
        
        **Built with ❤️ for Afghanistan** 🇦🇫
        
        *د افغانستان د پرمختګ لپاره جوړ شوی*
        """)
    
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666;">
    <p>🇦🇫 ZamAI Pro Voice Assistant v2.0 | Powered by Hugging Face | © 2025</p>
    </div>
    """)

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
