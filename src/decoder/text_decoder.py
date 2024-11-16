"""
Text Decoder Module for SemMomentSTT

This module handles the conversion of semantic trajectories into text:
- Trajectory to token mapping
- Context-aware decoding
- Word-level confidence scoring
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, NamedTuple
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn.functional as F
from ..semantic.types import SemanticTrajectory

class WordScore(NamedTuple):
    """Container for word-level scoring"""
    word: str
    confidence: float
    semantic_similarity: float
    language_model_score: float
    start_time: float
    duration: float

@dataclass
class DecodingResult:
    """Container for decoding results"""
    text: str
    confidence: float
    word_scores: List[WordScore]
    trajectory_id: int

class TextDecoder:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        min_confidence: float = 0.3,
        context_size: int = 5,
        lm_weight: float = 0.3,
        semantic_weight: float = 0.7,
        max_history: int = 100
    ):
        """
        Initialize the text decoder
        
        Args:
            model_name: Name of the pretrained language model to use
            device: Device to run the model on (cpu/cuda)
            min_confidence: Minimum confidence threshold for decoding
            context_size: Number of previous tokens to use for context
            lm_weight: Weight for language model score
            semantic_weight: Weight for semantic similarity score
            max_history: Maximum number of words to keep in history
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_confidence = min_confidence
        self.context_size = context_size
        self.lm_weight = lm_weight
        self.semantic_weight = semantic_weight
        self.max_history = max_history
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Cache for semantic-token mappings
        self.token_embeddings = self._initialize_token_embeddings()
        
        # Context management
        self.context_tokens: List[str] = []
        self.context_embeddings: List[np.ndarray] = []
        self.word_history: List[WordScore] = []
    
    def _initialize_token_embeddings(self) -> torch.Tensor:
        """
        Initialize embeddings for all tokens in vocabulary
        
        Returns:
            Normalized token embeddings tensor
        """
        with torch.no_grad():
            # Get embeddings for all tokens
            vocab_size = self.tokenizer.vocab_size
            token_ids = torch.arange(vocab_size).to(self.device)
            embeddings = self.model.get_input_embeddings()(token_ids)
            
            # Normalize embeddings and ensure float32
            return F.normalize(embeddings.float(), p=2, dim=1)
    
    def _find_nearest_tokens(
        self,
        semantic_vector: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find the k nearest tokens to a semantic vector
        
        Args:
            semantic_vector: Vector in semantic space
            k: Number of nearest neighbors to return
            
        Returns:
            List of (token, similarity) pairs
        """
        # Convert to tensor and normalize
        vector = torch.from_numpy(semantic_vector).float().to(self.device)
        vector = F.normalize(vector.unsqueeze(0), p=2, dim=1)
        
        # Compute similarities with all tokens
        similarities = torch.matmul(vector, self.token_embeddings.T).squeeze()
        
        # Get top-k tokens
        values, indices = torch.topk(similarities, k)
        
        return [
            (self.tokenizer.decode([idx.item()]), val.item())
            for idx, val in zip(indices, values)
        ]
    
    def _compute_language_model_scores(
        self,
        candidates: List[Tuple[str, float]],
        context: Optional[str] = None
    ) -> List[Tuple[str, float, float]]:
        """
        Compute language model scores for candidates
        
        Args:
            candidates: List of (token, similarity) pairs
            context: Optional context string
            
        Returns:
            List of (token, similarity, lm_score) tuples
        """
        if not candidates:
            return []
            
        context = context or " ".join(self.context_tokens[-self.context_size:])
        
        scored_candidates = []
        with torch.no_grad():
            for token, sim in candidates:
                # Create input with candidate
                text = f"{context} {token}"
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Get model prediction
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get score for the candidate token
                token_id = self.tokenizer.encode(token)[-1]
                lm_score = torch.softmax(
                    logits[0, -1],
                    dim=0
                )[token_id].item()
                
                scored_candidates.append((token, sim, lm_score))
        
        return scored_candidates
    
    def decode_trajectory(
        self,
        trajectory: SemanticTrajectory,
        timestamp: float,
        duration: float
    ) -> Optional[DecodingResult]:
        """
        Decode a semantic trajectory into text with word-level scoring
        
        Args:
            trajectory: Semantic trajectory to decode
            timestamp: Current timestamp in seconds
            duration: Duration of the current segment
            
        Returns:
            DecodingResult if successful, None otherwise
        """
        if trajectory.confidence < self.min_confidence:
            return None
        
        # Find nearest tokens
        candidates = self._find_nearest_tokens(trajectory.position)
        
        # Apply language model scoring
        scored_candidates = self._compute_language_model_scores(candidates)
        
        if scored_candidates:
            # Get best candidate
            word, semantic_sim, lm_score = scored_candidates[0]
            
            # Compute final confidence
            confidence = (
                self.semantic_weight * semantic_sim +
                self.lm_weight * lm_score
            ) * trajectory.confidence
            
            # Create word score
            word_score = WordScore(
                word=word,
                confidence=confidence,
                semantic_similarity=semantic_sim,
                language_model_score=lm_score,
                start_time=timestamp,
                duration=duration
            )
            
            # Update context
            self.context_tokens.append(word)
            self.context_embeddings.append(trajectory.position)
            
            # Maintain context size
            if len(self.context_tokens) > self.context_size:
                self.context_tokens.pop(0)
                self.context_embeddings.pop(0)
            
            # Update word history
            self.word_history.append(word_score)
            
            # Maintain history size
            if len(self.word_history) > self.max_history:
                self.word_history.pop(0)
            
            return DecodingResult(
                text=word,
                confidence=confidence,
                word_scores=[word_score],
                trajectory_id=trajectory.id
            )
        
        return None
    
    def get_word_history(
        self,
        time_window: Optional[float] = None
    ) -> List[WordScore]:
        """
        Get word history with optional time window
        
        Args:
            time_window: Optional time window in seconds
            
        Returns:
            List of word scores within the time window
        """
        if not self.word_history:
            return []
            
        if time_window is None:
            return self.word_history[-self.max_history:]
        
        latest_time = self.word_history[-1].start_time
        cutoff_time = latest_time - time_window
        
        # Filter by time window and respect max history size
        filtered_history = [
            score for score in self.word_history
            if score.start_time >= cutoff_time
        ]
        
        return filtered_history[-self.max_history:]
    
    def reset_context(self):
        """Reset the decoding context"""
        self.context_tokens.clear()
        self.context_embeddings.clear()
        self.word_history.clear()
    
    @property
    def context(self) -> str:
        """Get the current context string"""
        return " ".join(self.context_tokens)
