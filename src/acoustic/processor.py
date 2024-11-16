"""
Acoustic Processing Module for SemMomentSTT

This module handles the acoustic processing pipeline including:
- Feature extraction from audio input
- Wav2Vec2 model integration
- Sliding window processing
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn.functional as F

@dataclass
class AcousticFeatures:
    """Container for extracted acoustic features"""
    features: torch.Tensor
    timestamp: float
    window_size: int
    confidence: float

class AcousticProcessor:
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        device: Optional[str] = None,
        sample_rate: int = 16000,
        chunk_length: float = 0.5  # in seconds
    ):
        """
        Initialize the acoustic processor
        
        Args:
            model_name: Name of the pretrained Wav2Vec2 model
            device: Device to run the model on (cpu/cuda)
            sample_rate: Audio sample rate (default: 16kHz)
            chunk_length: Length of audio chunks to process (in seconds)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length
        self.chunk_samples = int(sample_rate * chunk_length)
        
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the Wav2Vec2 model and processor"""
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Wav2Vec2 model: {str(e)}")
    
    def _preprocess_audio(self, audio_frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio frame for Wav2Vec2 model
        
        Args:
            audio_frame: Raw audio frame data
            
        Returns:
            Preprocessed audio tensor
        """
        # Ensure correct shape and type
        if audio_frame.ndim == 1:
            audio_frame = audio_frame.reshape(1, -1)
            
        # Normalize if needed
        if audio_frame.dtype != np.float32:
            audio_frame = audio_frame.astype(np.float32) / np.iinfo(audio_frame.dtype).max
            
        # Process through Wav2Vec2 processor
        inputs = self.processor(
            audio_frame,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        return inputs.input_values.to(self.device)
    
    @torch.no_grad()
    def process_frame(self, audio_frame: np.ndarray) -> AcousticFeatures:
        """
        Process a single frame of audio
        
        Args:
            audio_frame: Raw audio frame data
            
        Returns:
            Extracted acoustic features
        """
        # Preprocess audio
        input_values = self._preprocess_audio(audio_frame)
        
        # Extract features through Wav2Vec2
        outputs = self.model(input_values)
        
        # Get the last hidden states
        features = outputs.last_hidden_state
        
        # Calculate confidence based on feature statistics
        confidence = torch.sigmoid(
            torch.mean(torch.std(features, dim=1))
        ).item()
        
        return AcousticFeatures(
            features=features,
            timestamp=len(audio_frame) / self.sample_rate,
            window_size=self.chunk_samples,
            confidence=confidence
        )
    
    def update_sliding_window(self, features: AcousticFeatures):
        """
        Update the sliding window with new features
        
        Args:
            features: New acoustic features to add
        """
        # TODO: Implement sliding window update for continuous processing
        pass
