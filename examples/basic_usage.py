"""
Basic usage examples for SemMomentSTT

This script demonstrates the core functionality of the SemMomentSTT system:
1. File transcription with various formats and sample rates
2. Microphone input with device configuration
3. Custom audio streaming
"""

import numpy as np
from pathlib import Path
import soundfile as sf
from src.main import SemMomentSTT

def print_trajectory(trajectory, prefix=""):
    """Helper function to print trajectory information"""
    print(f"{prefix}Position norm: {np.linalg.norm(trajectory.position):.3f}")
    print(f"{prefix}Confidence: {trajectory.confidence:.3f}")
    print(f"{prefix}State: {trajectory.state.value}")
    print()

def example_file_transcription():
    """Demonstrate file transcription with different formats"""
    print("\n=== File Transcription Example ===")
    
    stt = SemMomentSTT()
    
    # Example with different audio formats and sample rates
    audio_files = [
        ("path/to/audio1.wav", 16000),  # Standard WAV at 16kHz
        ("path/to/audio2.mp3", 44100),  # MP3 at 44.1kHz
        ("path/to/audio3.flac", 48000), # FLAC at 48kHz
    ]
    
    for audio_path, sample_rate in audio_files:
        if not Path(audio_path).exists():
            # Create a test file if none exists
            print(f"Creating test file at {sample_rate}Hz...")
            duration = 2.0  # seconds
            samples = int(duration * sample_rate)
            audio = np.random.randn(samples).astype(np.float32) * 0.1
            sf.write(audio_path, audio, sample_rate)
        
        print(f"\nProcessing: {audio_path} ({sample_rate}Hz)")
        try:
            trajectories = stt.transcribe_file(audio_path)
            print(f"Processed {len(trajectories)} segments")
            for i, trajectory in enumerate(trajectories[:3]):  # Show first 3
                print(f"\nSegment {i + 1}:")
                print_trajectory(trajectory, "  ")
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")

def example_microphone_input():
    """Demonstrate microphone input with device configuration"""
    print("\n=== Microphone Input Example ===")
    
    stt = SemMomentSTT()
    
    # Show available devices
    stt.list_audio_devices()
    
    # Example device configuration
    device_configs = [
        # Default device with default sample rate
        {"device": None, "input_sample_rate": None},
        # Specific device with custom sample rate
        # {"device": 1, "input_sample_rate": 44100},
    ]
    
    for config in device_configs:
        print(f"\nTrying device config: {config}")
        print("Speak into your microphone (Ctrl+C to stop)")
        
        try:
            for trajectory in stt.transcribe_microphone(**config):
                print("\nProcessed segment:")
                print_trajectory(trajectory, "  ")
        except KeyboardInterrupt:
            print("\nStopped microphone transcription")
        except Exception as e:
            print(f"Error with device config {config}: {str(e)}")

def example_custom_stream():
    """Demonstrate custom audio streaming with different sample rates"""
    print("\n=== Custom Stream Example ===")
    
    def create_audio_stream(sample_rate, duration=5.0, chunk_duration=0.5):
        """Generate audio frames at specified sample rate"""
        chunk_size = int(sample_rate * chunk_duration)
        total_chunks = int(duration / chunk_duration)
        
        for _ in range(total_chunks):
            # Generate chunk of audio
            chunk = np.random.randn(chunk_size).astype(np.float32) * 0.1
            yield chunk
    
    stt = SemMomentSTT()
    
    # Test with different sample rates
    sample_rates = [8000, 16000, 44100, 48000]
    
    for rate in sample_rates:
        print(f"\nProcessing stream at {rate}Hz...")
        stream = create_audio_stream(rate)
        
        try:
            for trajectory in stt.transcribe_stream(stream, stream_sample_rate=rate):
                print("\nProcessed segment:")
                print_trajectory(trajectory, "  ")
        except Exception as e:
            print(f"Error processing {rate}Hz stream: {str(e)}")

def main():
    """Run all examples"""
    print("SemMomentSTT Usage Examples")
    print("==========================")
    
    # Uncomment examples you want to run
    example_file_transcription()
    # example_microphone_input()
    example_custom_stream()

if __name__ == "__main__":
    main()
