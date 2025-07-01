#!/usr/bin/env python3
"""
Test the ZamAI-Whisper-v3-Pashto model
This script validates the functionality of the Pashto-optimized Whisper model
"""

import os
import torch
import argparse
import librosa
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np
import matplotlib.pyplot as plt
import json

def main():
    parser = argparse.ArgumentParser(description="Test ZamAI Whisper v3 Pashto model")
    parser.add_argument("--audio", "-a", help="Path to Pashto audio file")
    parser.add_argument("--model", "-m", default="tasal9/ZamAI-Whisper-v3-Pashto", 
                        help="Model ID (default: tasal9/ZamAI-Whisper-v3-Pashto)")
    parser.add_argument("--compare", "-c", action="store_true", 
                        help="Compare with base Whisper model")
    parser.add_argument("--language", "-l", default="ps", 
                        help="Language code (default: ps for Pashto)")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--generate_audio", "-g", action="store_true", 
                        help="Generate test audio if no input file")
    
    args = parser.parse_args()
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Generate test audio if requested and no audio file provided
    audio_path = args.audio
    if not audio_path and args.generate_audio:
        audio_path = "test_audio.wav"
        generate_test_audio(audio_path)
    
    if not audio_path:
        print("Error: No audio file provided. Use --audio or --generate_audio")
        return
    
    # Load audio
    try:
        print(f"Loading audio from {audio_path}")
        audio_data, sampling_rate = librosa.load(audio_path, sr=16000)
        duration = len(audio_data) / sampling_rate
        print(f"Audio duration: {duration:.2f} seconds")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
    
    # Test main model
    print(f"\n🔊 Testing model: {args.model}")
    try:
        # Load processor and model
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model).to(device)
        
        # Process audio
        input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        
        # Generate transcription
        print("Generating transcription...")
        start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        
        if start_time:
            start_time.record()
            
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features, 
                language=args.language, 
                task="transcribe"
            )
            
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        else:
            import time
            start = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    input_features=input_features, 
                    language=args.language, 
                    task="transcribe"
                )
            inference_time = time.time() - start
        
        # Decode transcription
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"✅ Transcription: {transcription}")
        print(f"⏱️ Inference time: {inference_time:.2f} seconds")
        
        results = {
            "model": args.model,
            "transcription": transcription,
            "inference_time": round(inference_time, 2),
            "language": args.language
        }
        
        # Compare with base model if requested
        if args.compare:
            print("\n🔄 Comparing with base model: openai/whisper-large-v3")
            
            base_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
            base_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3").to(device)
            
            # Process audio with base model
            base_input_features = base_processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            
            if device == "cuda":
                base_start = torch.cuda.Event(enable_timing=True)
                base_end = torch.cuda.Event(enable_timing=True)
                base_start.record()
            else:
                base_start_time = time.time()
            
            with torch.no_grad():
                base_generated_ids = base_model.generate(
                    input_features=base_input_features, 
                    language=args.language, 
                    task="transcribe"
                )
            
            if device == "cuda":
                base_end.record()
                torch.cuda.synchronize()
                base_inference_time = base_start.elapsed_time(base_end) / 1000
            else:
                base_inference_time = time.time() - base_start_time
            
            base_transcription = base_processor.batch_decode(base_generated_ids, skip_special_tokens=True)[0]
            print(f"✅ Base transcription: {base_transcription}")
            print(f"⏱️ Base inference time: {base_inference_time:.2f} seconds")
            
            results["comparison"] = {
                "base_model": "openai/whisper-large-v3",
                "base_transcription": base_transcription,
                "base_inference_time": round(base_inference_time, 2),
                "speed_improvement": round((base_inference_time - inference_time) / base_inference_time * 100, 2)
            }
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
    
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def generate_test_audio(output_path, duration=5, sample_rate=16000):
    """Generate a test audio file with sine waves at different frequencies"""
    print(f"Generating test audio file: {output_path}")
    
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a simple melody using sine waves
    frequencies = [261.63, 329.63, 392.00, 523.25]  # C4, E4, G4, C5
    durations = [1, 1, 1, 2]
    
    # Create the audio signal
    audio = np.zeros_like(t)
    start_idx = 0
    
    for freq, dur in zip(frequencies, durations):
        end_idx = start_idx + int(dur * sample_rate)
        if end_idx > len(t):
            end_idx = len(t)
        
        # Add sine wave with envelope
        segment = np.sin(2 * np.pi * freq * t[start_idx:end_idx])
        
        # Apply simple envelope
        envelope = np.ones_like(segment)
        attack = int(0.1 * sample_rate)
        release = int(0.2 * sample_rate)
        
        if len(segment) > attack:
            envelope[:attack] = np.linspace(0, 1, attack)
        if len(segment) > release:
            envelope[-release:] = np.linspace(1, 0, release)
            
        segment = segment * envelope
        audio[start_idx:end_idx] += segment
        start_idx = end_idx
    
    # Normalize audio
    audio = 0.5 * audio / np.max(np.abs(audio))
    
    # Save audio file
    sf.write(output_path, audio, sample_rate)
    print(f"✅ Test audio generated: {duration}s duration, {sample_rate}Hz sample rate")
    
    # Plot waveform
    plt.figure(figsize=(10, 4))
    plt.plot(t, audio)
    plt.title("Test Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path.replace('.wav', '_waveform.png')
    plt.savefig(plot_path)
    print(f"✅ Waveform plot saved: {plot_path}")

if __name__ == "__main__":
    main()
