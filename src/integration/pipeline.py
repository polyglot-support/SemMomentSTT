"""
Integration Layer for SemMomentSTT

This module handles the integration between acoustic and semantic components:
- Acoustic-semantic mapping
- Context integration
- Pipeline coordination
- Text decoding with word-level confidence
- N-best list and lattice generation
"""

import torch
import numpy as np
from typing import Optional, List, Generator, Tuple, NamedTuple
import torch.nn.functional as F

from ..acoustic.processor import AcousticProcessor, AcousticFeatures
from ..semantic.momentum_tracker import MomentumTracker, SemanticTrajectory
from ..semantic.lattice import WordLattice, LatticePath
from ..decoder.text_decoder import TextDecoder, WordScore, DecodingResult

class NBestHypothesis(NamedTuple):
    """Container for N-best hypothesis"""
    text: str
    confidence: float
    word_scores: List[WordScore]
    trajectory_path: List[SemanticTrajectory]

class ProcessingResult(NamedTuple):
    """Container for processing results"""
    trajectory: Optional[SemanticTrajectory]
    decoding_result: Optional[DecodingResult]
    n_best: List[NBestHypothesis]
    lattice_paths: List[LatticePath]
    features: Optional[AcousticFeatures] = None

class IntegrationPipeline:
    def __init__(
        self,
        acoustic_model: str = "facebook/wav2vec2-base",
        language_model: str = "bert-base-uncased",
        semantic_dim: int = 768,
        context_window: int = 10,
        device: Optional[str] = None,
        max_trajectories: int = 5,
        n_best: int = 5,
        force_scale: float = 10.0,  # Increased for stronger semantic forces
        step_size: float = 50.0,  # Reduced for smoother movement
        momentum_decay: float = 0.999,  # Increased for better momentum
        min_confidence: float = 0.1,
        merge_threshold: float = 0.85
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
            n_best: Number of hypotheses to maintain
            force_scale: Scaling factor for semantic forces
            step_size: Step size for momentum updates
            momentum_decay: Decay factor for momentum
            min_confidence: Minimum confidence threshold
            merge_threshold: Threshold for merging trajectories
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.context_window = context_window
        self.semantic_dim = semantic_dim
        self.n_best = n_best
        
        # Initialize components
        self.acoustic_processor = AcousticProcessor(
            model_name=acoustic_model,
            device=self.device
        )
        
        self.momentum_tracker = MomentumTracker(
            semantic_dim=semantic_dim,
            max_trajectories=max_trajectories,
            beam_width=n_best,
            force_scale=force_scale,
            momentum_decay=momentum_decay,
            min_confidence=min_confidence,
            merge_threshold=merge_threshold
        )
        
        self.text_decoder = TextDecoder(
            model_name=language_model,
            device=self.device
        )
        
        self.word_lattice = WordLattice()
        
        # Context buffer
        self.context_buffer: List[AcousticFeatures] = []
        
        # Time tracking
        self.current_time: float = 0.0
        
        # Initialize semantic projection layer using BERT embeddings
        self.semantic_projection = self._initialize_semantic_projection()
    
    def _initialize_semantic_projection(self) -> torch.nn.Module:
        """Initialize semantic projection using BERT embeddings"""
        with torch.no_grad():
            # Get BERT embeddings
            bert_embeddings = self.text_decoder.model.get_input_embeddings()
            bert_dim = bert_embeddings.weight.shape[1]
            
            # Create projection layer
            projection = torch.nn.Sequential(
                torch.nn.Linear(
                    self.acoustic_processor.model.config.hidden_size,
                    bert_dim
                ),
                torch.nn.LayerNorm(bert_dim),
                torch.nn.GELU(),
                torch.nn.Linear(bert_dim, self.semantic_dim)
            ).to(self.device)
            
            # Initialize first layer with PCA-like projection
            acoustic_dim = self.acoustic_processor.model.config.hidden_size
            U, _, _ = torch.linalg.svd(
                bert_embeddings.weight.float(),
                full_matrices=False
            )
            if acoustic_dim > bert_dim:
                init_weight = U.T[:bert_dim, :]
            else:
                init_weight = U.T[:acoustic_dim, :bert_dim]
            
            projection[0].weight.data.copy_(init_weight)
            
            return projection
    
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
        
        # Get context embedding
        context_embedding = self.get_context_embedding()
        
        # Map to semantic space with context
        semantic_vector = self._map_to_semantic_space(
            acoustic_features.features,
            context_embedding
        )
        
        # Update semantic trajectories
        self.momentum_tracker.update_trajectories(
            semantic_vector,
            confidence=acoustic_features.confidence
        )
        
        # Get trajectory paths
        trajectory_paths = self.momentum_tracker.get_trajectory_paths()
        
        # Process paths and build lattice
        n_best_results = []
        lattice_paths = []
        best_decoding = None
        best_trajectory = None
        
        if trajectory_paths:
            # Collect word scores for each path
            path_word_scores = []
            
            for path in trajectory_paths:
                path_scores = []
                for traj in path:
                    # Get word scores for trajectory
                    decoding = self.text_decoder.decode_trajectory(
                        traj,
                        timestamp=acoustic_features.timestamp,
                        duration=frame_duration
                    )
                    
                    if decoding is not None:
                        word_score = decoding.word_scores[0]
                        path_scores.append((
                            decoding.text,
                            word_score.confidence,
                            word_score.language_model_score,
                            word_score.semantic_similarity
                        ))
                        
                        # Keep track of best result
                        if (best_decoding is None or 
                            decoding.confidence > best_decoding.confidence):
                            best_decoding = decoding
                            best_trajectory = traj
                
                if path_scores:  # Only add paths with valid scores
                    path_word_scores.append(path_scores)
            
            if path_word_scores:  # Only build lattice if we have valid scores
                # Build lattice from paths
                self.word_lattice.build_from_trajectories(
                    trajectory_paths,
                    path_word_scores
                )
                
                # Get N-best paths from lattice
                lattice_paths = self.word_lattice.find_best_paths(
                    n_paths=self.n_best
                )
                
                # Convert lattice paths to N-best hypotheses
                for path in lattice_paths:
                    # Find corresponding trajectory path
                    traj_path = []
                    for node in path.nodes:
                        traj = next(
                            (t for t in trajectory_paths[0]  # Use first path as reference
                            if t.id == node.trajectory_id),
                            None
                        )
                        if traj is not None:
                            traj_path.append(traj)
                    
                    if traj_path:  # Only add hypotheses with valid trajectories
                        n_best_results.append(NBestHypothesis(
                            text=" ".join(node.word for node in path.nodes),
                            confidence=path.total_score,
                            word_scores=[
                                WordScore(
                                    word=node.word,
                                    confidence=node.confidence,
                                    semantic_similarity=path.semantic_score,
                                    language_model_score=path.language_score,
                                    start_time=node.timestamp,
                                    duration=frame_duration
                                )
                                for node in path.nodes
                            ],
                            trajectory_path=traj_path
                        ))
        
        # Update time tracking
        self.current_time += frame_duration
        
        features = acoustic_features if return_features else None
        return ProcessingResult(
            best_trajectory,
            best_decoding,
            n_best_results,
            lattice_paths,
            features
        )
    
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
    
    def _map_to_semantic_space(
        self,
        features: torch.Tensor,
        context_embedding: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Map acoustic features to semantic space
        
        Args:
            features: Acoustic features tensor
            context_embedding: Optional context embedding
            
        Returns:
            Vector in semantic space
        """
        # Ensure input is a tensor
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(self.device)
        
        # Average pooling over time dimension
        pooled_features = torch.mean(features, dim=1)  # [batch_size, hidden_size]
        
        # Project to semantic space
        with torch.no_grad():
            semantic_vector = self.semantic_projection(pooled_features)
            
            # Add context influence if available
            if context_embedding is not None:
                context_tensor = torch.from_numpy(context_embedding).to(self.device)
                semantic_vector = 0.7 * semantic_vector + 0.3 * context_tensor
            
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
            self._map_to_semantic_space(f.features)
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
            List of word scores with timing information
        """
        return self.text_decoder.get_word_history(time_window)
    
    def get_lattice_dot(self) -> str:
        """
        Get DOT representation of current lattice
        
        Returns:
            DOT format string for visualization
        """
        return self.word_lattice.to_dot()
    
    def reset(self):
        """Reset the pipeline state"""
        self.context_buffer.clear()
        self.text_decoder.reset_context()
        self.current_time = 0.0
        self.acoustic_processor.reset()
        self.momentum_tracker = MomentumTracker(  # Reset momentum tracker
            semantic_dim=self.semantic_dim,
            max_trajectories=self.n_best,
            beam_width=self.n_best,
            force_scale=10.0,
            momentum_decay=0.999,
            min_confidence=0.1,
            merge_threshold=0.85
        )
