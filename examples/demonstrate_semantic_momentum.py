"""
Demonstration of Semantic Momentum in Speech Recognition

This script demonstrates the key features of semantic momentum:
1. Downloads LibriSpeech samples if not present
2. Shows trajectory tracking with semantic momentum
3. Visualizes competing hypotheses and force fields
4. Demonstrates disambiguation using semantic context
"""

import os
import requests
import tarfile
from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.main import SemMomentSTT

def download_librispeech_sample():
    """Download a small sample from LibriSpeech dev-clean"""
    sample_dir = Path("samples")
    sample_dir.mkdir(exist_ok=True)
    
    # Download specific file known to have interesting semantic content
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    sample_path = sample_dir / "dev-clean.tar.gz"
    
    if not sample_path.exists():
        print("Downloading LibriSpeech sample...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(sample_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print("Extracting sample...")
        with tarfile.open(sample_path) as tar:
            # Extract only one speaker's samples
            members = [m for m in tar.getmembers() 
                      if "LibriSpeech/dev-clean/84/121123/" in m.name]
            tar.extractall(sample_dir, members=members)
        
        print("Sample downloaded and extracted.")

def demonstrate_semantic_momentum():
    """Demonstrate semantic momentum features"""
    print("\nSemantic Momentum Demonstration")
    print("==============================")
    
    # Initialize STT system with visualization enabled
    stt = SemMomentSTT(
        n_best=3,
        semantic_dim=768,
        force_scale=10.0,
        momentum_decay=0.999
    )
    
    # Process sample files
    sample_dir = Path("samples/LibriSpeech/dev-clean/84/121123")
    if not sample_dir.exists():
        print("Downloading samples first...")
        download_librispeech_sample()
    
    # Get list of flac files
    audio_files = list(sample_dir.glob("*.flac"))
    
    for audio_path in audio_files:
        print(f"\nProcessing: {audio_path.name}")
        
        # Get detailed transcription with trajectory analysis
        results = stt.transcribe_file(
            audio_path,
            return_word_scores=True,
            return_n_best=True
        )
        
        print("\nTranscription Analysis:")
        for i, result in enumerate(results):
            print(f"\nSegment {i+1}:")
            
            # Show competing hypotheses
            print("\nCompeting Hypotheses:")
            for j, hyp in enumerate(result.n_best):
                print(f"\nHypothesis {j+1} ({hyp.confidence*100:.1f}% confidence):")
                print(f"Text: {hyp.text}")
                
                # Show word-level analysis
                print("Word Analysis:")
                for word_score in hyp.word_scores:
                    print(f"  {word_score.word:<15} "
                          f"Semantic: {word_score.semantic_similarity*100:4.1f}% "
                          f"Language: {word_score.language_model_score*100:4.1f}%")
            
            # Visualize semantic trajectory
            plt.figure(figsize=(12, 6))
            
            # Plot trajectory in 2D projection
            trajectory_path = result.n_best[0].trajectory_path
            if trajectory_path:
                from sklearn.decomposition import PCA
                positions = [t.position for t in trajectory_path]
                
                if len(positions) > 1:
                    pca = PCA(n_components=2)
                    trajectory_2d = pca.fit_transform(positions)
                    
                    plt.subplot(1, 2, 1)
                    plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'b-', label='Semantic Path')
                    plt.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], c='g', label='Start')
                    plt.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], c='r', label='End')
                    
                    # Add word labels at trajectory points
                    for point, word in zip(trajectory_2d, hyp.word_scores):
                        plt.annotate(word.word, (point[0], point[1]))
                    
                    plt.title('Semantic Trajectory')
                    plt.legend()
            
            # Visualize word lattice
            plt.subplot(1, 2, 2)
            lattice_dot = stt.get_lattice_visualization()
            # Note: You would need graphviz to render the DOT format
            # For now, just show the competing paths textually
            plt.text(0.1, 0.5, '\n'.join(
                f"Path {i+1}: {' -> '.join(node.word for node in path.nodes)}"
                for i, path in enumerate(result.lattice_paths)
            ), fontsize=8)
            plt.title('Word Lattice')
            plt.axis('off')
            
            # Save visualization
            plt.savefig(f'trajectory_{audio_path.stem}_{i}.png')
            plt.close()
            
            print(f"\nVisualization saved as trajectory_{audio_path.stem}_{i}.png")

def main():
    """Run the demonstration"""
    demonstrate_semantic_momentum()

if __name__ == "__main__":
    main()
