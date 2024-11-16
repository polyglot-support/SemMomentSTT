"""
Integration Layer for SemMomentSTT

This module handles the integration between acoustic and semantic components:
- Acoustic-semantic mapping
- Context integration
- Pipeline coordination
- Text decoding with word-level confidence
"""

from typing import Optional, List, Generator, Tuple, NamedTuple
import numpy as np
import torch
import torch.nn.functional as F

from ..acoustic.processor import AcousticProcessor, AcousticFeatures
from ..semantic.momentum_tracker import MomentumTracker, SemanticTrajectory
from ..decoder.text_decoder import TextDecoder, WordScore, DecodingResult

class ProcessingResult(NamedTuple):
    """Container for processing results"""
    trajectory: Optional[SemanticTrajectory]
    decoding_result: Optional[DecodingResult]
    features: Optional[AcousticFeatures] = None

class IntegrationPipeline:
    def __init__(
        self,
        acoustic_model: str = "facebook/wav2vec2-base",
        language_model: str = "bert-base-uncased",
        semantic_dim: int = 768,
        context_window: int = 10,
        device: Optional[str] = None,
        max_trajectories: int = 5
    ):
        """
        Initialize the integration pipeline
        
        Args:
            acoustic_model: Name of the acoustic model to use
            language_model: Name of the language model for decoding
            semantic_dim: Dimensionality of semantic space
            context_window: Size of context window in frames
            device: Device to run models on
            max_trajectories: Maximum number of semantic trajectories
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.context_window = context_window
        self.semantic_dim = semantic_dim
        
        # Initialize components
        self.acoustic_processor = AcousticProcessor(
            model_name=acoustic_model,
            device=self.device
        )
        
        self.momentum_tracker = MomentumTracker(
            semantic_dim=semantic_dim,
            max_trajectories=max_trajectories
        )
        
        self.text_decoder = TextDecoder(
            model_name=language_model,
            device=self.device
        )
        
        # Context buffer
        self.context_buffer: List[AcousticFeatures] = []
        
        # Time tracking
        self.current_time: float = 0.0
        
        # Initialize semantic projection layer
        self.semantic_projection = torch.nn.Linear(
            self.acoustic_processor.model.config.hidden_size,
            semantic_dim
        ).to(self.device)
    
    def process_frame(
        self,
        audio_frame: np.ndarray,
        orig_sr: Optional[int] = None,
        return_features: bool = False,
        frame_duration: Optional[float] = None
    ) -> ProcessingResult:
        """
        Process a single frame of audio through the full pipeline
        
        Args:
            audio_frame: Raw audio frame data
            orig_sr: Original sample rate if different from target
            return_features: Whether to return acoustic features
            frame_duration: Duration of the frame in seconds
            
        Returns:
            ProcessingResult containing trajectory and decoded text
        """
        # Extract acoustic features
        acoustic_features = self.acoustic_processor.process_frame(
            audio_frame,
            orig_sr=orig_sr
        )
        
        # Calculate frame duration if not provided
        if frame_duration is None:
            frame_duration = len(audio_frame) / (
                orig_sr or self.acoustic_processor.sample_rate
            )
        
        # Update context
        self._update_context(acoustic_features)
        
        # Map to semantic space
        semantic_vector = self._map_to_semantic_space(acoustic_features)
        
        # Update semantic trajectories
        self.momentum_tracker.update_trajectories(
            semantic_vector,
            confidence=acoustic_features.confidence
        )
        
        # Maintain trajectories
        self.momentum_tracker.merge_similar_trajectories()
        self.momentum_tracker.prune_trajectories()
        
        # Get best trajectory
        trajectory = self.momentum_tracker.get_best_trajectory()
        
        # Decode text if trajectory exists
        decoding_result = None
        if trajectory is not None:
            decoding_result = self.text_decoder.decode_trajectory(
                trajectory,
                timestamp=self.current_time,
                duration=frame_duration
            )
        
        # Update time tracking
        self.current_time += frame_duration
        
        features = acoustic_features if return_features else None
        return ProcessingResult(trajectory, decoding_result, features)
    
    def process_stream(
        self,
        audio_stream: Generator[np.ndarray, None, None],
        stream_sr: Optional[int] = None,
        frame_duration: Optional[float] = None
    ) -> Generator[ProcessingResult, None, None]:
        """
        Process a stream of audio frames
        
        Args:
            audio_stream: Generator yielding audio frames
            stream_sr: Sample rate of the audio stream
            frame_duration: Duration of each frame in seconds
            
        Yields:
            ProcessingResult for each processed frame
        """
        for audio_frame in audio_stream:
            result = self.process_frame(
                audio_frame,
                orig_sr=stream_sr,
                frame_duration=frame_duration
            )
            yield result
    
    def _update_context(self, features: AcousticFeatures):
        """
        Update the context buffer with new features
        
        Args:
            features: New acoustic features to add
        """
        self.context_buffer.append(features)
        if len(self.context_buffer) > self.context_window:
            self.context_buffer.pop(0)
    
    def _map_to_semantic_space(self, features: AcousticFeatures) -> np.ndarray:
        """
        Map acoustic features to semantic space
        
        Args:
            features: Acoustic features to map
            
        Returns:
            Vector in semantic space
        """
        # Average pooling over time dimension
        pooled_features = torch.mean(features.features, dim=1)
        
        # Project to semantic space
        with torch.no_grad():
            semantic_vector = self.semantic_projection(pooled_features)
            semantic_vector = F.normalize(semantic_vector, p=2, dim=1)
        
        # Convert to numpy and ensure correct shape
        return semantic_vector.squeeze().cpu().numpy()
    
    def get_context_embedding(self) -> Optional[np.ndarray]:
        """
        Get the current context embedding
        
        Returns:
            Context embedding if available
        """
        if not self.context_buffer:
            return None
            
        # Combine context features with attention to confidence
        confidences = torch.tensor([
            f.confidence for f in self.context_buffer
        ]).to(self.device)
        
        # Softmax over confidences
        weights = F.softmax(confidences, dim=0)
        
        # Weighted average of semantic vectors
        context_vectors = [
            self._map_to_semantic_space(f)
            for f in self.context_buffer
        ]
        context_vectors = torch.stack([
            torch.from_numpy(v).to(self.device)
            for v in context_vectors
        ])
        
        weighted_sum = torch.sum(
            context_vectors * weights.unsqueeze(1),
            dim=0
        )
        
        return weighted_sum.cpu().numpy()
    
    def get_word_history(
        self,
        time_window: Optional[float] = None
    ) -> List[WordScore]:
        """
        Get word history with optional time window
        
        Args:
            time_window: Optional time window in seconds
            
        Returns:
            List of word scores
        """
        return self.text_decoder.get_word_history(time_window)
    
    def reset(self):
        """Reset the pipeline state"""
        self.context_buffer.clear()
        self.text_decoder.reset_context()
        self.current_time = 0.0
        # Note: momentum tracker state persists intentionally
