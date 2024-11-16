"""
Semantic Momentum Tracker for SemMomentSTT

This module implements the core semantic momentum tracking functionality:
- Multi-trajectory state maintenance
- Momentum vector computation
- Semantic force field modeling
- Confidence scoring
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
from scipy.spatial.distance import cosine
import torch
import torch.nn.functional as F

class TrajectoryState(Enum):
    ACTIVE = "active"
    MERGED = "merged"
    PRUNED = "pruned"

@dataclass
class SemanticTrajectory:
    """Represents a single semantic trajectory"""
    id: int
    position: np.ndarray  # Position in semantic space
    momentum: np.ndarray  # Current momentum vector
    confidence: float
    state: TrajectoryState
    history: List[np.ndarray]
    
    def update_position(self, force: np.ndarray, dt: float = 1.0):
        """Update position based on current momentum and force"""
        self.momentum = self.momentum + force * dt
        self.position = self.position + self.momentum * dt
        self.history.append(self.position.copy())

class MomentumTracker:
    def __init__(
        self,
        semantic_dim: int = 768,
        max_trajectories: int = 5,
        momentum_decay: float = 0.95,
        min_confidence: float = 0.1,
        merge_threshold: float = 0.85,
        force_scale: float = 1.0
    ):
        """
        Initialize the momentum tracker
        
        Args:
            semantic_dim: Dimensionality of the semantic space
            max_trajectories: Maximum number of active trajectories
            momentum_decay: Decay factor for momentum
            min_confidence: Minimum confidence threshold for pruning
            merge_threshold: Cosine similarity threshold for merging trajectories
            force_scale: Scaling factor for semantic forces
        """
        self.semantic_dim = semantic_dim
        self.max_trajectories = max_trajectories
        self.momentum_decay = momentum_decay
        self.min_confidence = min_confidence
        self.merge_threshold = merge_threshold
        self.force_scale = force_scale
        
        self.trajectories: Dict[int, SemanticTrajectory] = {}
        self.next_trajectory_id = 0
        
        # Initialize attraction centers (could be learned or updated)
        self.attraction_centers = np.random.randn(10, semantic_dim)
        self.attraction_strengths = np.ones(10)
    
    def compute_force_field(self, position: np.ndarray) -> np.ndarray:
        """
        Compute the semantic force field at a given position
        
        Args:
            position: Position in semantic space
            
        Returns:
            Force vector at the given position
        """
        # Initialize force vector
        force = np.zeros_like(position)
        
        # Add forces from attraction centers
        for center, strength in zip(self.attraction_centers, self.attraction_strengths):
            direction = center - position
            distance = np.linalg.norm(direction)
            if distance > 0:
                # Force decreases with square of distance
                force += strength * direction / (distance ** 2)
        
        # Add forces from other trajectories
        for trajectory in self.active_trajectories:
            if not np.array_equal(trajectory.position, position):
                direction = trajectory.position - position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    # Repulsive force from other trajectories
                    force -= direction / (distance ** 3)
        
        return force * self.force_scale
    
    def update_trajectories(self, acoustic_evidence: np.ndarray, confidence: float):
        """
        Update all active trajectories based on new evidence
        
        Args:
            acoustic_evidence: New acoustic evidence vector
            confidence: Confidence score for the new evidence
        """
        # Create new trajectory if needed
        if len(self.active_trajectories) < self.max_trajectories:
            self._create_trajectory(acoustic_evidence, confidence)
        
        # Update existing trajectories
        for trajectory in self.active_trajectories:
            # Apply semantic forces
            force = self.compute_force_field(trajectory.position)
            
            # Add force from acoustic evidence
            evidence_force = acoustic_evidence - trajectory.position
            force += evidence_force * confidence
            
            # Update trajectory
            trajectory.update_position(force)
            
            # Apply momentum decay
            trajectory.momentum *= self.momentum_decay
            
            # Update confidence based on consistency and evidence
            self._update_confidence(trajectory, acoustic_evidence, confidence)
    
    def _create_trajectory(self, position: np.ndarray, confidence: float):
        """Create a new trajectory"""
        trajectory = SemanticTrajectory(
            id=self.next_trajectory_id,
            position=position.copy(),
            momentum=np.zeros_like(position),
            confidence=confidence,
            state=TrajectoryState.ACTIVE,
            history=[position.copy()]
        )
        self.trajectories[self.next_trajectory_id] = trajectory
        self.next_trajectory_id += 1
    
    def _update_confidence(
        self,
        trajectory: SemanticTrajectory,
        evidence: np.ndarray,
        evidence_confidence: float
    ):
        """Update trajectory confidence based on new evidence"""
        # Compute similarity with evidence
        similarity = 1 - cosine(trajectory.position, evidence)
        
        # Update confidence as weighted average
        trajectory.confidence = (
            0.7 * trajectory.confidence +
            0.3 * similarity * evidence_confidence
        )
    
    def merge_similar_trajectories(self):
        """Merge trajectories that are close in semantic space"""
        active = self.active_trajectories
        merged_ids = set()
        
        for i, t1 in enumerate(active):
            if t1.id in merged_ids:
                continue
                
            for t2 in active[i+1:]:
                if t2.id in merged_ids:
                    continue
                    
                similarity = 1 - cosine(t1.position, t2.position)
                if similarity > self.merge_threshold:
                    # Merge t2 into t1
                    weight1 = t1.confidence / (t1.confidence + t2.confidence)
                    weight2 = t2.confidence / (t1.confidence + t2.confidence)
                    
                    t1.position = weight1 * t1.position + weight2 * t2.position
                    t1.momentum = weight1 * t1.momentum + weight2 * t2.momentum
                    t1.confidence = max(t1.confidence, t2.confidence)
                    
                    t2.state = TrajectoryState.MERGED
                    merged_ids.add(t2.id)
    
    def prune_trajectories(self):
        """Remove low confidence trajectories"""
        for trajectory in self.active_trajectories:
            if trajectory.confidence < self.min_confidence:
                trajectory.state = TrajectoryState.PRUNED
    
    @property
    def active_trajectories(self) -> List[SemanticTrajectory]:
        """Get list of active trajectories"""
        return [
            t for t in self.trajectories.values()
            if t.state == TrajectoryState.ACTIVE
        ]
    
    def get_best_trajectory(self) -> Optional[SemanticTrajectory]:
        """
        Get the highest confidence trajectory
        
        Returns:
            The trajectory with highest confidence, if any exist
        """
        active = self.active_trajectories
        if not active:
            return None
        return max(active, key=lambda t: t.confidence)
