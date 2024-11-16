"""
Word Lattice Visualization Example

This script demonstrates the word lattice functionality of SemMomentSTT:
1. Lattice construction from transcription
2. Path analysis and scoring
3. DOT format visualization
4. Real-time lattice updates
"""

import numpy as np
from pathlib import Path
import soundfile as sf
import graphviz
from src.main import SemMomentSTT
from src.semantic.lattice import LatticePath

def format_time(seconds: float) -> str:
    """Format time in seconds to MM:SS.mmm"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"

def format_path(path: LatticePath) -> str:
    """Format lattice path information"""
    lines = [
        f"Text: {' '.join(node.word for node in path.nodes)}",
        f"Total Score: {path.total_score*100:.1f}%",
        "Component Scores:",
        f"  Acoustic: {path.acoustic_score*100:.1f}%",
        f"  Language: {path.language_score*100:.1f}%",
        f"  Semantic: {path.semantic_score*100:.1f}%",
        "Word Details:"
    ]
    
    for node in path.nodes:
        lines.append(
            f"  {node.word:<15} "
            f"[{format_time(node.timestamp)}] "
            f"Conf: {node.confidence*100:.1f}%"
        )
    
    return "\n".join(lines)

def save_lattice_visualization(dot_str: str, output_path: str):
    """Save lattice visualization as image"""
    graph = graphviz.Source(dot_str)
    graph.render(output_path, format='png', cleanup=True)

def example_file_transcription():
    """Demonstrate lattice construction from file transcription"""
    print("\n=== File Transcription with Lattice ===")
    
    stt = SemMomentSTT(n_best=5)  # Keep top 5 hypotheses
    
    # Example with different audio formats
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
            # Get detailed transcription with lattice paths
            results = stt.transcribe_file(
                audio_path,
                return_word_scores=True,
                return_n_best=True,
                return_lattice=True
            )
            
            print("\nTranscription Results:")
            for i, result in enumerate(results):
                print(f"\nSegment {i+1} [{format_time(result.timestamp)}]")
                
                if result.lattice_paths:
                    print("\nLattice Paths:")
                    for j, path in enumerate(result.lattice_paths):
                        print(f"\nPath {j+1}:")
                        print(format_path(path))
                
                # Save lattice visualization
                dot = stt.get_lattice_visualization()
                output_path = f"lattice_segment_{i+1}"
                save_lattice_visualization(dot, output_path)
                print(f"\nLattice visualization saved to {output_path}.png")
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")

def example_real_time_lattice():
    """Demonstrate real-time lattice updates"""
    print("\n=== Real-time Lattice Visualization ===")
    
    stt = SemMomentSTT(n_best=3)
    
    # Show available devices
    stt.list_audio_devices()
    
    print("\nStarting microphone transcription...")
    print("Speak into your microphone (Ctrl+C to stop)")
    
    try:
        for i, result in enumerate(stt.transcribe_microphone(
            return_n_best=True,
            return_lattice=True
        )):
            print(f"\n[{format_time(result.timestamp)}]")
            
            if result.lattice_paths:
                print("\nCurrent Lattice Paths:")
                for j, path in enumerate(result.lattice_paths):
                    print(f"\nPath {j+1}:")
                    print(format_path(path))
                
                # Save lattice visualization periodically
                if i % 10 == 0:  # Every 10 frames
                    dot = stt.get_lattice_visualization()
                    output_path = f"lattice_frame_{i}"
                    save_lattice_visualization(dot, output_path)
                    print(f"\nLattice visualization saved to {output_path}.png")
    
    except KeyboardInterrupt:
        print("\nStopped transcription")

def example_lattice_analysis():
    """Demonstrate detailed lattice analysis"""
    print("\n=== Lattice Analysis Example ===")
    
    def create_audio_stream(sample_rate=16000, duration=5.0, chunk_duration=0.5):
        """Generate audio frames"""
        chunk_size = int(sample_rate * chunk_duration)
        total_chunks = int(duration / chunk_duration)
        
        for _ in range(total_chunks):
            yield np.random.randn(chunk_size).astype(np.float32) * 0.1
    
    stt = SemMomentSTT(n_best=5)
    stream = create_audio_stream()
    
    print("\nProcessing audio stream...")
    try:
        for i, result in enumerate(stt.transcribe_stream(
            stream,
            return_n_best=True,
            return_lattice=True
        )):
            print(f"\nFrame {i+1} [{format_time(result.timestamp)}]")
            
            if result.lattice_paths:
                # Analyze path scores
                print("\nPath Analysis:")
                scores = {
                    'acoustic': [],
                    'language': [],
                    'semantic': []
                }
                
                for path in result.lattice_paths:
                    scores['acoustic'].append(path.acoustic_score)
                    scores['language'].append(path.language_score)
                    scores['semantic'].append(path.semantic_score)
                
                # Show score distributions
                for score_type, values in scores.items():
                    avg = np.mean(values)
                    std = np.std(values)
                    print(f"\n{score_type.title()} Scores:")
                    print(f"  Average: {avg*100:.1f}%")
                    print(f"  Std Dev: {std*100:.1f}%")
                    print("  Distribution:")
                    for val in values:
                        print(f"    {'='*int(val*40):40s} {val*100:.1f}%")
                
                # Save visualization
                if i % 5 == 0:  # Every 5 frames
                    dot = stt.get_lattice_visualization()
                    output_path = f"lattice_analysis_{i}"
                    save_lattice_visualization(dot, output_path)
                    print(f"\nLattice visualization saved to {output_path}.png")
    
    except Exception as e:
        print(f"Error in analysis: {str(e)}")

def main():
    """Run all examples"""
    print("SemMomentSTT Lattice Examples")
    print("============================")
    
    # Uncomment examples you want to run
    example_file_transcription()
    # example_real_time_lattice()
    example_lattice_analysis()

if __name__ == "__main__":
    main()
