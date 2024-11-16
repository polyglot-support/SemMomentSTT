"""
Acoustic Processing Module for SemMomentSTT

This module handles the acoustic feature extraction:
- Audio preprocessing
- Feature extraction
- Sample rate conversion
"""

import torch
import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
import librosa
import torch.nn.functional as F

# Filter specific Wav2Vec2 initialization warnings
warnings.filterwarnings("ignore", message="Some weights of Wav2Vec2Model were not initialized from the model checkpoint*")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model on a down-stream task*")

@dataclass
class AcousticFeatures:
    """Container for acoustic features"""
    features: torch.Tensor  # Shape: (batch, time, features)
    confidence: float
    timestamp: float  # Time in seconds from start

class AcousticProcessor:
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        device: Optional[str] = None,
        sample_rate: int = 16000,
        chunk_length: float = 0.5,
        min_confidence: float = 0.1
    ):
        """
        Initialize the acoustic processor
        
        Args:
            model_name: Name of the pretrained model to use
            device: Device to run the model on (cpu/cuda)
            sample_rate: Target sample rate for audio
            chunk_length: Length of audio chunks in seconds
            min_confidence: Minimum confidence threshold
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length
        self.model_name = model_name
        self.min_confidence = min_confidence
        
        # Calculate chunk size in samples
        self.chunk_samples = int(chunk_length * sample_rate)
        
        # Initialize model and processor with warnings suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message="Some weights of Wav2Vec2Model were not initialized from the model checkpoint*")
            warnings.filterwarnings("ignore", message="You should probably TRAIN this model on a down-stream task*")
            self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model.eval()
        
        # Get model config
        self.config = self.model.config
        
        # Current timestamp
        self.current_time = 0.0
        
        # Initialize feature statistics for normalization
        self.feature_mean = None
        self.feature_std = None
    
    def _preprocess_audio(
        self,
        audio: np.ndarray,
        orig_sr: Optional[int] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Preprocess audio data
        
        Args:
            audio: Raw audio data
            orig_sr: Original sample rate if different from target
            
        Returns:
            Tuple of (preprocessed audio tensor, duration in seconds)
        """
        # Calculate duration before any processing
        duration = len(audio) / (orig_sr or self.sample_rate)
        
        # Resample if needed
        if orig_sr is not None and orig_sr != self.sample_rate:
            audio = self._resample_audio(audio, orig_sr)
        
        # Convert to float32 and normalize
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Normalize using RMS energy
        rms = np.sqrt(np.mean(emphasized_audio**2))
        if rms > 0:
            emphasized_audio = emphasized_audio / rms
        
        # Convert to tensor and add batch dimension
        audio_tensor = torch.from_numpy(emphasized_audio).unsqueeze(0).to(self.device)
        
        return audio_tensor, duration
    
    def _resample_audio(
        self,
        audio: np.ndarray,
        orig_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio: Audio data to resample
            orig_sr: Original sample rate
            
        Returns:
            Resampled audio data
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return librosa.resample(
                y=audio,
                orig_sr=orig_sr,
                target_sr=self.sample_rate
            )
    
    def _compute_confidence(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute confidence score from features
        
        Args:
            features: Feature tensor
            attention_mask: Optional attention mask
            
        Returns:
            Confidence score in [0, 1] range
        """
        # Compute feature statistics
        feature_norms = torch.norm(features, dim=-1)  # [batch, time]
        
        if attention_mask is not None:
            # Use attention mask to focus on valid regions
            masked_norms = feature_norms * attention_mask
            valid_positions = attention_mask.sum()
            if valid_positions > 0:
                mean_norm = masked_norms.sum() / valid_positions
            else:
                mean_norm = 0.0
        else:
            mean_norm = feature_norms.mean()
        
        # Compute temporal consistency
        if features.size(1) > 1:
            temporal_diff = torch.norm(
                features[:, 1:] - features[:, :-1],
                dim=-1
            ).mean()
            temporal_consistency = torch.exp(-temporal_diff)
        else:
            temporal_consistency = 1.0
        
        # Combine magnitude and consistency scores
        confidence = (0.7 * (mean_norm / 10.0) + 0.3 * temporal_consistency).item()
        
        # Clip to valid range
        return float(min(max(confidence, self.min_confidence), 1.0))
    
    def _normalize_features(
        self,
        features: torch.Tensor,
        update_stats: bool = True
    ) -> torch.Tensor:
        """
        Normalize features using running statistics
        
        Args:
            features: Feature tensor to normalize
            update_stats: Whether to update running statistics
            
        Returns:
            Normalized features
        """
        # Initialize statistics if needed
        if self.feature_mean is None:
            self.feature_mean = torch.zeros(features.size(-1)).to(self.device)
            self.feature_std = torch.ones(features.size(-1)).to(self.device)
        
        # Update running statistics
        if update_stats:
            with torch.no_grad():
                batch_mean = features.mean(dim=(0, 1))
                batch_std = features.std(dim=(0, 1))
                
                # Update using exponential moving average
                momentum = 0.1
                self.feature_mean = (1 - momentum) * self.feature_mean + momentum * batch_mean
                self.feature_std = (1 - momentum) * self.feature_std + momentum * batch_std
        
        # Normalize features
        normalized = (features - self.feature_mean) / (self.feature_std + 1e-5)
        
        return normalized
    
    def process_frame(
        self,
        audio_frame: np.ndarray,
        orig_sr: Optional[int] = None
    ) -> AcousticFeatures:
        """
        Process a single frame of audio
        
        Args:
            audio_frame: Raw audio frame data
            orig_sr: Original sample rate if different from target
            
        Returns:
            Extracted acoustic features
        """
        # Calculate duration before any processing
        duration = len(audio_frame) / (orig_sr or self.sample_rate)
        
        # Get current timestamp before updating
        timestamp = self.current_time
        
        # Update timestamp before processing to match expected behavior
        self.current_time += duration
        
        # Preprocess audio
        audio_tensor, _ = self._preprocess_audio(audio_frame, orig_sr)
        
        # Process with wav2vec2
        inputs = self.processor(
            audio_tensor.squeeze().cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(
                inputs.input_values.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device) if hasattr(inputs, 'attention_mask') else None
            )
            features = outputs.last_hidden_state
            
            # Ensure features have batch dimension
            if features.dim() == 2:
                features = features.unsqueeze(0)
            
            # Get attention mask
            attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, 'attention_mask') else None
            
            # Normalize features
            features = self._normalize_features(features)
            
            # Apply attention mask if available
            if attention_mask is not None:
                attention_mask = F.interpolate(
                    attention_mask.float().unsqueeze(1),
                    size=features.size(1),
                    mode='nearest'
                ).squeeze(1)
                features = features * attention_mask.unsqueeze(-1)
            
            # Compute confidence score
            confidence = self._compute_confidence(features, attention_mask)
        
        return AcousticFeatures(
            features=features,
            confidence=confidence,
            timestamp=timestamp
        )
    
    def reset(self):
        """Reset processor state"""
        self.current_time = 0.0
        self.feature_mean = None
        self.feature_std = None
    
    @property
    def AcousticFeatures(self):
        """Expose AcousticFeatures class for type checking"""
        return AcousticFeatures
