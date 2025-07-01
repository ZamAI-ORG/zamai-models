import gradio as gr
from api_client import HFInferenceClient
import os

class ZamAIVoiceAssistant:
    def __init__(self):
        token = os.getenv("HUGGINGFACE_TOKEN", "")
        self.hf_client = HFInferenceClient(token=token)
        
    def process_audio(self, audio):
        """STT → LLM → TTS workflow"""
        try:
            # Speech to text
            text = self.hf_client.stt_process(audio, model="tasal9/ZamAI-STT-Processor")
            
            # Generate response
            prompt = f"User (Pashto): {text}\nAssistant (Pashto):"
            response = self.hf_client.generate_text(
                prompt=prompt, 
                model="tasal9/ZamAI-Voice-Assistant"
            )
            
            # Text to speech
            audio_out = self.hf_client.tts_process(response, model="tasal9/ZamAI-TTS-Generator")
            
            return response, audio_out
            
        except Exception as e:
            return f"Error: {str(e)}", None

# Initialize assistant
assistant = ZamAIVoiceAssistant()

# Gradio interface
demo = gr.Interface(
    fn=assistant.process_audio,
    inputs=gr.Audio(source="microphone", type="filepath", label="🎤 Speak in Pashto"),
    outputs=[
        gr.Textbox(label="📝 Response Text", lines=3),
        gr.Audio(label="🔊 Voice Response")
    ],
    title="🇦🇫 ZamAI Voice Assistant",
    description="Speak in Pashto and get AI-powered responses",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)
