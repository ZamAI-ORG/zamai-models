#!/usr/bin/env python3
"""
Speech-to-Text Processor for ZamAI
Optimized for Pashto language recognition
"""
import os
import tempfile
from pathlib import Path
import numpy as np
import torch

class SpeechToTextProcessor:
    """
    Advanced speech-to-text processing with Pashto optimization
    Leverages Hugging Face Pro for enhanced performance
    """
    
    def __init__(self, client, model_id="tasal9/Pashto-Whisper-Large-Pro"):
        """
        Initialize the STT processor
        
        Args:
            client: HFInferenceClient object
            model_id: Model ID for speech recognition
        """
        self.client = client
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"STT Processor initialized with model: {self.model_id}")
        print(f"Using device: {self.device}")
        
        # Dialect-specific configuration
        self.dialect_configs = {
            "ps": {"task": "transcribe", "language": "ps"},
            "ps-AF": {"task": "transcribe", "language": "ps", "dialect": "afghan"},
            "ps-PK": {"task": "transcribe", "language": "ps", "dialect": "pakistani"}
        }

    def preprocess_audio(self, audio_path):
        """
        Preprocess audio for optimal recognition
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            Processed audio data
        """
        # Convert audio to the correct format if needed
        # Implement noise reduction for better Pashto recognition
        return audio_path

    def transcribe(self, audio_path, language="ps"):
        """
        Transcribe Pashto speech to text
        
        Args:
            audio_path: Path to the audio file
            language: Language/dialect code
        
        Returns:
            Transcribed text
        """
        try:
            # Get dialect-specific configuration
            config = self.dialect_configs.get(language, self.dialect_configs["ps"])
            
            # Process the audio file
            processed_audio = self.preprocess_audio(audio_path)
            
            # Use the client to transcribe
            result = self.client.stt_process(
                audio=processed_audio,
                model=self.model_id,
                **config
            )
            
            # Post-process the transcription (fix common Pashto recognition issues)
            transcription = self.postprocess_text(result.get("text", ""))
            
            return transcription
            
        except Exception as e:
            print(f"Error in transcription: {e}")
            return "د غږ پېژندنې په وخت کې تخنیکي ستونزه رامنځته شوه."  # Technical issue occurred during voice recognition
    
    def postprocess_text(self, text):
        """
        Post-process the transcribed text
        
        Args:
            text: Raw transcription
        
        Returns:
            Cleaned and corrected transcription
        """
        # Implement Pashto-specific post-processing:
        # 1. Fix common recognition errors for Pashto
        # 2. Handle dialectical variations
        # 3. Normalize text representation
        
        return text
