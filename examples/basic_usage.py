"""
Basic usage examples for SemMomentSTT

This script demonstrates the core functionality of the SemMomentSTT system:
1. File transcription
2. Microphone input
3. Custom audio streaming
"""

import numpy as np
from pathlib import Path
from src.main import SemMomentSTT

def print_trajectory(trajectory, prefix=""):
    """Helper function to print trajectory information"""
    print(f"{prefix}Position norm: {np.linalg.norm(trajectory.position):.3f}")
    print(f"{prefix}Confidence: {trajectory.confidence:.3f}")
    print(f"{prefix}State: {trajectory.state.value}")
    print()

def example_file_transcription():
    """Demonstrate file transcription"""
    print("\n=== File Transcription Example ===")
    
    stt = SemMomentSTT()
    
    # Replace with your audio file path
    audio_path = "path/to/your/audio.wav"
    
    if not Path(audio_path).exists():
        print(f"Please provide a valid audio file at {audio_path}")
        return
    
    print(f"Transcribing file: {audio_path}")
    trajectories = stt.transcribe_file(audio_path)
    
    print(f"\nProcessed {len(trajectories)} segments")
    for i, trajectory in enumerate(trajectories[:5]):  # Show first 5
        print(f"\nSegment {i + 1}:")
        print_trajectory(trajectory, "  ")

def example_microphone_input():
    """Demonstrate microphone input"""
    print("\n=== Microphone Input Example ===")
    
    stt = SemMomentSTT()
    
    # Show available devices
    stt.list_audio_devices()
    
    print("\nStarting microphone transcription...")
    print("Speak into your microphone (Ctrl+C to stop)")
    
    try:
        for trajectory in stt.transcribe_microphone():
            print("\nProcessed segment:")
            print_trajectory(trajectory, "  ")
    except KeyboardInterrupt:
        print("\nStopped microphone transcription")

def example_custom_stream():
    """Demonstrate custom audio streaming"""
    print("\n=== Custom Stream Example ===")
    
    def dummy_audio_stream():
        """Generate dummy audio frames for demonstration"""
        # Generate 5 seconds of audio in 0.5s chunks
        chunk_size = 8000  # 0.5s at 16kHz
        for _ in range(10):  # 5 seconds total
            yield np.random.randn(chunk_size).astype(np.float32)
    
    stt = SemMomentSTT()
    
    print("Processing custom audio stream...")
    for trajectory in stt.transcribe_stream(dummy_audio_stream()):
        print("\nProcessed segment:")
        print_trajectory(trajectory, "  ")

def main():
    """Run all examples"""
    print("SemMomentSTT Usage Examples")
    print("==========================")
    
    # Uncomment examples you want to run
    # example_file_transcription()
    # example_microphone_input()
    example_custom_stream()

if __name__ == "__main__":
    main()
