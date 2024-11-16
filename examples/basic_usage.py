"""
Basic usage examples for SemMomentSTT

This script demonstrates the core functionality of the SemMomentSTT system:
1. File transcription with word-level confidence scores
2. Real-time microphone transcription with detailed word timing
3. Custom audio streaming with word-level analysis
"""

import numpy as np
from pathlib import Path
import soundfile as sf
from src.main import SemMomentSTT

def format_time(seconds: float) -> str:
    """Format time in seconds to MM:SS.mmm"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"

def format_word_score(word_score) -> str:
    """Format word score information"""
    return (
        f"{word_score.word:<15} "
        f"Conf: {word_score.confidence*100:4.1f}% "
        f"(Sem: {word_score.semantic_similarity*100:4.1f}%, "
        f"LM: {word_score.language_model_score*100:4.1f}%)"
    )

def example_file_transcription():
    """Demonstrate file transcription with word-level analysis"""
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
            # Get detailed transcription with word scores
            results = stt.transcribe_file(
                audio_path,
                return_word_scores=True
            )
            
            print("\nDetailed transcription:")
            for result in results:
                print(f"\n[{format_time(result.timestamp)}] "
                      f"Overall confidence: {result.confidence*100:.1f}%")
                
                for word_score in result.word_scores:
                    print(f"  {format_word_score(word_score)}")
            
            # Get simple transcription
            text = stt.transcribe_file(audio_path)
            print(f"\nFull text: {text}")
            
            # Get word history for the last second
            recent_words = stt.get_word_history(time_window=1.0)
            print("\nRecent word history (last 1 second):")
            for word_score in recent_words:
                print(f"  [{format_time(word_score.start_time)}] "
                      f"{format_word_score(word_score)}")
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")

def example_microphone_input():
    """Demonstrate real-time microphone transcription with word analysis"""
    print("\n=== Microphone Transcription Example ===")
    
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
            for result in stt.transcribe_microphone(**config):
                print(f"\n[{format_time(result.timestamp)}] "
                      f"Overall confidence: {result.confidence*100:.1f}%")
                
                for word_score in result.word_scores:
                    print(f"  {format_word_score(word_score)}")
        except KeyboardInterrupt:
            print("\nStopped microphone transcription")
        except Exception as e:
            print(f"Error with device config {config}: {str(e)}")
        finally:
            stt.reset()  # Reset system state between configurations

def example_custom_stream():
    """Demonstrate custom audio streaming with word-level analysis"""
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
            for result in stt.transcribe_stream(
                stream,
                stream_sample_rate=rate,
                chunk_duration=0.5
            ):
                print(f"\n[{format_time(result.timestamp)}] "
                      f"Overall confidence: {result.confidence*100:.1f}%")
                
                for word_score in result.word_scores:
                    print(f"  {format_word_score(word_score)}")
                    
                # Show semantic similarity vs language model contribution
                word = word_score.word
                sem_conf = word_score.semantic_similarity
                lm_conf = word_score.language_model_score
                print(f"  Analysis for '{word}':")
                print(f"    Semantic: {'='*int(sem_conf*40):40s} {sem_conf*100:4.1f}%")
                print(f"    Language: {'='*int(lm_conf*40):40s} {lm_conf*100:4.1f}%")
                
        except Exception as e:
            print(f"Error processing {rate}Hz stream: {str(e)}")
        finally:
            stt.reset()  # Reset system state between sample rates

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
