"""
Text Decoder Module for SemMomentSTT

This module handles the conversion of semantic trajectories into text:
- Trajectory to token mapping
- Context-aware decoding
- Confidence-based filtering
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForMaskedLM
from ..semantic.momentum_tracker import SemanticTrajectory

class TextDecoder:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        min_confidence: float = 0.3,
        context_size: int = 5
    ):
        """
        Initialize the text decoder
        
        Args:
            model_name: Name of the pretrained language model to use
            device: Device to run the model on (cpu/cuda)
            min_confidence: Minimum confidence threshold for decoding
            context_size: Number of previous tokens to use for context
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_confidence = min_confidence
        self.context_size = context_size
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Cache for semantic-token mappings
        self.token_embeddings = self._initialize_token_embeddings()
        
        # Context management
        self.context_tokens: List[str] = []
        self.context_embeddings: List[np.ndarray] = []
    
    def _initialize_token_embeddings(self) -> torch.Tensor:
        """Initialize embeddings for all tokens in vocabulary"""
        with torch.no_grad():
            # Get embeddings for all tokens
            vocab_size = self.tokenizer.vocab_size
            token_ids = torch.arange(vocab_size).to(self.device)
            embeddings = self.model.get_input_embeddings()(token_ids)
            
            # Normalize embeddings
            return F.normalize(embeddings, p=2, dim=1)
    
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
        vector = torch.from_numpy(semantic_vector).to(self.device)
        vector = F.normalize(vector.unsqueeze(0), p=2, dim=1)
        
        # Compute similarities with all tokens
        similarities = torch.matmul(vector, self.token_embeddings.T).squeeze()
        
        # Get top-k tokens
        values, indices = torch.topk(similarities, k)
        
        return [
            (self.tokenizer.decode([idx.item()]), val.item())
            for idx, val in zip(indices, values)
        ]
    
    def _apply_language_model(
        self,
        candidates: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Apply language model scoring to candidate tokens
        
        Args:
            candidates: List of (token, similarity) pairs
            
        Returns:
            Rescored candidates
        """
        if not self.context_tokens:
            return candidates
        
        # Create context string
        context = " ".join(self.context_tokens[-self.context_size:])
        
        # Score each candidate
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
                score = torch.softmax(
                    logits[0, -1],
                    dim=0
                )[token_id].item()
                
                # Combine similarity with language model score
                final_score = 0.7 * sim + 0.3 * score
                scored_candidates.append((token, final_score))
        
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)
    
    def decode_trajectory(
        self,
        trajectory: SemanticTrajectory
    ) -> Optional[Tuple[str, float]]:
        """
        Decode a semantic trajectory into text
        
        Args:
            trajectory: Semantic trajectory to decode
            
        Returns:
            Tuple of (decoded text, confidence) if successful
        """
        if trajectory.confidence < self.min_confidence:
            return None
        
        # Find nearest tokens
        candidates = self._find_nearest_tokens(trajectory.position)
        
        # Apply language model scoring
        scored_candidates = self._apply_language_model(candidates)
        
        if scored_candidates:
            best_token, score = scored_candidates[0]
            
            # Update context
            self.context_tokens.append(best_token)
            self.context_embeddings.append(trajectory.position)
            
            # Maintain context size
            if len(self.context_tokens) > self.context_size:
                self.context_tokens.pop(0)
                self.context_embeddings.pop(0)
            
            return best_token, score * trajectory.confidence
        
        return None
    
    def reset_context(self):
        """Reset the decoding context"""
        self.context_tokens.clear()
        self.context_embeddings.clear()
    
    @property
    def context(self) -> str:
        """Get the current context string"""
        return " ".join(self.context_tokens)
