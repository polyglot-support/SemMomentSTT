"""
Main module for SemMomentSTT

Provides high-level interface for the semantic momentum-based speech recognition system.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Optional, Generator, Union, List, Tuple
from queue import Queue
from threading import Thread, Event
import torch
import librosa

from .integration.pipeline import IntegrationPipeline
from .semantic.momentum_tracker import SemanticTrajectory

class SemMomentSTT:
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        semantic_dim: int = 768,
        device: Optional[str] = None,
        sample_rate: int = 16000
    ):
        """
        Initialize the speech recognition system
        
        Args:
            model_name: Name of the acoustic model to use
            semantic_dim: Dimensionality of semantic space
            device: Device to run on (cpu/cuda)
            sample_rate: Target audio sample rate in Hz
        """
        self.pipeline = IntegrationPipeline(
            acoustic_model=model_name,
            semantic_dim=semantic_dim,
            device=device
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
        chunk_duration: float = 0.5
    ) -> List[SemanticTrajectory]:
        """
        Transcribe an audio file
        
        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each audio chunk in seconds
            
        Returns:
            List of semantic trajectories
        """
        # Load audio file
        audio, file_sample_rate = self._load_audio(audio_path)
        
        # Process in chunks
        chunk_size = int(file_sample_rate * chunk_duration)
        trajectories = []
        
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
            trajectory = self.pipeline.process_frame(
                chunk,
                orig_sr=file_sample_rate
            )
            
            if trajectory is not None:
                trajectories.append(trajectory)
        
        return trajectories
    
    def transcribe_stream(
        self,
        audio_stream: Generator[np.ndarray, None, None],
        stream_sample_rate: Optional[int] = None
    ) -> Generator[SemanticTrajectory, None, None]:
        """
        Transcribe a stream of audio data
        
        Args:
            audio_stream: Generator yielding audio frames
            stream_sample_rate: Sample rate of the audio stream
            
        Yields:
            Semantic trajectories as they become available
        """
        for audio_frame in audio_stream:
            trajectory = self.pipeline.process_frame(
                audio_frame,
                orig_sr=stream_sample_rate
            )
            if trajectory is not None:
                yield trajectory
    
    def transcribe_microphone(
        self,
        device: Optional[int] = None,
        chunk_duration: float = 0.5,
        input_sample_rate: Optional[int] = None
    ) -> Generator[SemanticTrajectory, None, None]:
        """
        Transcribe audio from microphone in real-time
        
        Args:
            device: Audio input device ID (None for default)
            chunk_duration: Duration of each audio chunk in seconds
            input_sample_rate: Sample rate to use for input (None for device default)
            
        Yields:
            Semantic trajectories as they become available
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
                    trajectory = self.pipeline.process_frame(
                        audio_chunk.flatten(),
                        orig_sr=stream_sr
                    )
                    
                    if trajectory is not None:
                        yield trajectory
                        
            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                self._stop_recording.set()
    
    def stop_microphone(self):
        """Stop microphone transcription"""
        self._stop_recording.set()
    
    @staticmethod
    def list_audio_devices() -> None:
        """Print available audio input devices"""
        print("\nAvailable audio input devices:")
        print(sd.query_devices())
    
    @property
    def device(self) -> str:
        """Get the current device being used"""
        return self.pipeline.device
