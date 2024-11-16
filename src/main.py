"""
Main module for SemMomentSTT

Provides high-level interface for the semantic momentum-based speech recognition system.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Optional, Generator, Union, List
from queue import Queue
from threading import Thread, Event
import torch

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
            sample_rate: Audio sample rate in Hz
        """
        self.pipeline = IntegrationPipeline(
            acoustic_model=model_name,
            semantic_dim=semantic_dim,
            device=device
        )
        self.sample_rate = sample_rate
        self._stop_recording = Event()
    
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
        audio, file_sample_rate = sf.read(audio_path)
        
        # Resample if needed
        if file_sample_rate != self.sample_rate:
            # TODO: Implement resampling
            raise ValueError(
                f"Sample rate mismatch: {file_sample_rate} != {self.sample_rate}"
            )
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Process in chunks
        chunk_size = int(self.sample_rate * chunk_duration)
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
            
            trajectory = self.pipeline.process_frame(chunk)
            if trajectory is not None:
                trajectories.append(trajectory)
        
        return trajectories
    
    def transcribe_stream(
        self,
        audio_stream: Generator[np.ndarray, None, None]
    ) -> Generator[SemanticTrajectory, None, None]:
        """
        Transcribe a stream of audio data
        
        Args:
            audio_stream: Generator yielding audio frames
            
        Yields:
            Semantic trajectories as they become available
        """
        return self.pipeline.process_stream(audio_stream)
    
    def transcribe_microphone(
        self,
        device: Optional[int] = None,
        chunk_duration: float = 0.5
    ) -> Generator[SemanticTrajectory, None, None]:
        """
        Transcribe audio from microphone in real-time
        
        Args:
            device: Audio input device ID (None for default)
            chunk_duration: Duration of each audio chunk in seconds
            
        Yields:
            Semantic trajectories as they become available
        """
        chunk_size = int(self.sample_rate * chunk_duration)
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
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_size,
            device=device,
            callback=audio_callback
        ):
            print("Listening... Press Ctrl+C to stop.")
            try:
                while not self._stop_recording.is_set():
                    # Get audio chunk from queue
                    audio_chunk = audio_queue.get()
                    
                    # Process audio chunk
                    trajectory = self.pipeline.process_frame(
                        audio_chunk.flatten()
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
