"""
Main module for SemMomentSTT

Provides high-level interface for the semantic momentum-based speech recognition system.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Optional, Generator, Union, List, Tuple, NamedTuple
from queue import Queue
from threading import Thread, Event
import torch
import librosa

from .integration.pipeline import (
    IntegrationPipeline,
    ProcessingResult,
    NBestHypothesis
)
from .semantic.lattice import LatticePath
from .decoder.text_decoder import WordScore

class TranscriptionResult(NamedTuple):
    """Container for transcription results"""
    text: Optional[str]
    confidence: Optional[float]
    timestamp: float
    word_scores: Optional[List[WordScore]] = None
    n_best: Optional[List[NBestHypothesis]] = None
    lattice_paths: Optional[List[LatticePath]] = None

class SemMomentSTT:
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        language_model: str = "bert-base-uncased",
        semantic_dim: int = 768,
        device: Optional[str] = None,
        sample_rate: int = 16000,
        n_best: int = 5
    ):
        """
        Initialize the speech recognition system
        
        Args:
            model_name: Name of the acoustic model to use
            language_model: Name of the language model for decoding
            semantic_dim: Dimensionality of semantic space
            device: Device to run on (cpu/cuda)
            sample_rate: Target audio sample rate in Hz
            n_best: Number of hypotheses to maintain
        """
        self.pipeline = IntegrationPipeline(
            acoustic_model=model_name,
            language_model=language_model,
            semantic_dim=semantic_dim,
            device=device,
            n_best=n_best
        )
        self.sample_rate = sample_rate
        self._stop_recording = Event()
    
    def _load_audio(
        self,
        audio_path: Union[str, Path]
    ) -> Tuple[np.ndarray, int]:
        """
        Load and prepare audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio data, sample rate)
        """
        try:
            # Try soundfile first (faster for wav files)
            audio, file_sample_rate = sf.read(audio_path)
        except:
            # Fall back to librosa (handles more formats)
            audio, file_sample_rate = librosa.load(
                audio_path,
                sr=None  # Preserve original sample rate
            )
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        return audio, file_sample_rate
    
    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        chunk_duration: float = 0.5,
        return_word_scores: bool = False,
        return_n_best: bool = False,
        return_lattice: bool = False
    ) -> Union[str, List[TranscriptionResult]]:
        """
        Transcribe an audio file
        
        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each audio chunk in seconds
            return_word_scores: Whether to return detailed word-level results
            return_n_best: Whether to return N-best hypotheses
            return_lattice: Whether to return lattice paths
            
        Returns:
            Transcribed text or list of TranscriptionResult with details
        """
        # Load audio file
        audio, file_sample_rate = self._load_audio(audio_path)
        
        # Process in chunks
        chunk_size = int(file_sample_rate * chunk_duration)
        results = []
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(
                    chunk,
                    (0, chunk_size - len(chunk)),
                    mode='constant'
                )
            
            # Process chunk with original sample rate
            result = self.pipeline.process_frame(
                chunk,
                orig_sr=file_sample_rate,
                frame_duration=chunk_duration
            )
            
            if result.decoding_result is not None:
                results.append(TranscriptionResult(
                    text=result.decoding_result.text,
                    confidence=result.decoding_result.confidence,
                    timestamp=result.decoding_result.word_scores[0].start_time,
                    word_scores=result.decoding_result.word_scores if return_word_scores else None,
                    n_best=result.n_best if return_n_best else None,
                    lattice_paths=result.lattice_paths if return_lattice else None
                ))
        
        if return_word_scores or return_n_best or return_lattice:
            return results
        else:
            # Join text with spaces, filtering None values
            return " ".join(r.text for r in results if r.text is not None)
    
    def transcribe_stream(
        self,
        audio_stream: Generator[np.ndarray, None, None],
        stream_sample_rate: Optional[int] = None,
        chunk_duration: Optional[float] = None,
        return_n_best: bool = False,
        return_lattice: bool = False
    ) -> Generator[TranscriptionResult, None, None]:
        """
        Transcribe a stream of audio data
        
        Args:
            audio_stream: Generator yielding audio frames
            stream_sample_rate: Sample rate of the audio stream
            chunk_duration: Duration of each chunk in seconds
            return_n_best: Whether to return N-best hypotheses
            return_lattice: Whether to return lattice paths
            
        Yields:
            TranscriptionResult for each processed chunk
        """
        for audio_frame in audio_stream:
            result = self.pipeline.process_frame(
                audio_frame,
                orig_sr=stream_sample_rate,
                frame_duration=chunk_duration
            )
            
            if result.decoding_result is not None:
                yield TranscriptionResult(
                    text=result.decoding_result.text,
                    confidence=result.decoding_result.confidence,
                    timestamp=result.decoding_result.word_scores[0].start_time,
                    word_scores=result.decoding_result.word_scores,
                    n_best=result.n_best if return_n_best else None,
                    lattice_paths=result.lattice_paths if return_lattice else None
                )
    
    def transcribe_microphone(
        self,
        device: Optional[int] = None,
        chunk_duration: float = 0.5,
        input_sample_rate: Optional[int] = None,
        return_n_best: bool = False,
        return_lattice: bool = False
    ) -> Generator[TranscriptionResult, None, None]:
        """
        Transcribe audio from microphone in real-time
        
        Args:
            device: Audio input device ID (None for default)
            chunk_duration: Duration of each audio chunk in seconds
            input_sample_rate: Sample rate to use for input (None for device default)
            return_n_best: Whether to return N-best hypotheses
            return_lattice: Whether to return lattice paths
            
        Yields:
            TranscriptionResult as text becomes available
        """
        # Get device info
        if device is not None:
            device_info = sd.query_devices(device, 'input')
            device_sr = int(device_info['default_samplerate'])
        else:
            device_sr = 44100  # Common default
        
        # Use specified rate or device default
        stream_sr = input_sample_rate or device_sr
        chunk_size = int(stream_sr * chunk_duration)
        
        audio_queue = Queue()
        self._stop_recording.clear()
        
        def audio_callback(indata, frames, time, status):
            """Callback for audio input"""
            if status:
                print(f"Audio input status: {status}")
            if not self._stop_recording.is_set():
                audio_queue.put(indata.copy())
        
        # Start audio input stream
        with sd.InputStream(
            samplerate=stream_sr,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_size,
            device=device,
            callback=audio_callback
        ):
            print(f"Listening... (Input rate: {stream_sr}Hz) Press Ctrl+C to stop.")
            try:
                while not self._stop_recording.is_set():
                    # Get audio chunk from queue
                    audio_chunk = audio_queue.get()
                    
                    # Process audio chunk
                    result = self.pipeline.process_frame(
                        audio_chunk.flatten(),
                        orig_sr=stream_sr,
                        frame_duration=chunk_duration
                    )
                    
                    if result.decoding_result is not None:
                        yield TranscriptionResult(
                            text=result.decoding_result.text,
                            confidence=result.decoding_result.confidence,
                            timestamp=result.decoding_result.word_scores[0].start_time,
                            word_scores=result.decoding_result.word_scores,
                            n_best=result.n_best if return_n_best else None,
                            lattice_paths=result.lattice_paths if return_lattice else None
                        )
                        
            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                self._stop_recording.set()
    
    def get_word_history(
        self,
        time_window: Optional[float] = None
    ) -> List[WordScore]:
        """
        Get word history with optional time window
        
        Args:
            time_window: Optional time window in seconds
            
        Returns:
            List of word scores with timing information
        """
        return self.pipeline.get_word_history(time_window)
    
    def get_lattice_visualization(self) -> str:
        """
        Get DOT format visualization of current word lattice
        
        Returns:
            DOT format string for visualization
        """
        return self.pipeline.get_lattice_dot()
    
    def stop_microphone(self):
        """Stop microphone transcription"""
        self._stop_recording.set()
    
    def reset(self):
        """Reset the system state"""
        self.pipeline.reset()
    
    @staticmethod
    def list_audio_devices() -> None:
        """Print available audio input devices"""
        print("\nAvailable audio input devices:")
        print(sd.query_devices())
    
    @property
    def device(self) -> str:
        """Get the current device being used"""
        return self.pipeline.device
