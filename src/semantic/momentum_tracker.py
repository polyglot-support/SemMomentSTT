"""
Semantic Momentum Tracker for SemMomentSTT

This module implements the core semantic momentum tracking functionality:
- Multi-trajectory state maintenance
- Momentum vector computation
- Semantic force field modeling
- Beam search for trajectory management
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import cosine
import torch
import torch.nn.functional as F

from .types import SemanticTrajectory, TrajectoryState, BeamHypothesis
from .beam_search import BeamSearch

class MomentumTracker:
    def __init__(
        self,
        semantic_dim: int = 768,
        max_trajectories: int = 5,
        momentum_decay: float = 0.95,  # Keep as is
        min_confidence: float = 0.1,
        merge_threshold: float = 0.85,  # Keep as is
        force_scale: float = 1.274,  # Keep as is
        beam_width: int = 3,
        beam_depth: int = 5
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
            beam_width: Width of beam search
            beam_depth: Maximum depth of beam search
        """
        self.semantic_dim = semantic_dim
        self.max_trajectories = max_trajectories
        self.momentum_decay = momentum_decay
        self.min_confidence = min_confidence
        self.merge_threshold = merge_threshold
        self.force_scale = force_scale
        
        # Initialize beam search
        self.beam_search = BeamSearch(
            beam_width=beam_width,
            max_depth=beam_depth,
            score_threshold=min_confidence,
            diversity_penalty=0.1
        )
        
        self.trajectories: Dict[int, SemanticTrajectory] = {}
        self.next_trajectory_id = 0
        
        # Initialize attraction centers
        self.attraction_centers = np.random.randn(10, semantic_dim)
        self.attraction_centers = self.attraction_centers / np.linalg.norm(
            self.attraction_centers, axis=1, keepdims=True
        )
        self.attraction_strengths = np.ones(10) * 0.05
    
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
            distance = max(np.linalg.norm(direction), 1e-6)
            force += strength * direction / (1 + distance ** 2)
        
        # Add forces from other trajectories
        for trajectory in self.active_trajectories:
            if not np.array_equal(trajectory.position, position):
                direction = trajectory.position - position
                distance = max(np.linalg.norm(direction), 1e-6)
                force -= 0.01 * direction / (1 + distance ** 2)
        
        # Scale and clip force
        force = np.clip(force * self.force_scale, -1.0, 1.0)
        
        return force
    
    def update_trajectories(self, acoustic_evidence: np.ndarray, confidence: float):
        """
        Update all active trajectories based on new evidence
        
        Args:
            acoustic_evidence: New acoustic evidence vector
            confidence: Confidence score for the new evidence
        """
        # Make a copy of the evidence vector
        acoustic_evidence = np.array(acoustic_evidence, copy=True)
        
        # Check if we can merge with any existing trajectory
        merged = False
        if self.active_trajectories:
            for traj in self.active_trajectories:
                similarity = 1 - cosine(traj.position, acoustic_evidence)
                if similarity > self.merge_threshold:
                    # Merge new evidence into existing trajectory
                    weight1 = traj.confidence / (traj.confidence + confidence)
                    weight2 = confidence / (traj.confidence + confidence)
                    traj.position = weight1 * traj.position + weight2 * acoustic_evidence
                    traj.confidence = max(traj.confidence, confidence)
                    merged = True
                    break
        
        # Create new trajectory if not merged
        if not merged and len(self.active_trajectories) < self.beam_search.beam_width:
            trajectory = self._create_trajectory(acoustic_evidence, confidence)
            self.trajectories[trajectory.id] = trajectory
            return  # Return early to avoid updating the new trajectory
        
        # Update existing trajectories
        for traj in self.active_trajectories:
            # First update - apply force to build momentum
            if np.allclose(traj.momentum, 0):
                # Compute initial force
                force = acoustic_evidence - traj.position
                force = np.clip(force * self.force_scale, -1.0, 1.0)
                
                # Update momentum with step size
                traj.momentum = force * 0.01  
                
                # Update position
                traj.position += traj.momentum
            else:
                # Apply momentum decay first
                traj.momentum *= self.momentum_decay
                
                # Apply semantic forces
                force = self.compute_force_field(traj.position)
                
                # Add force from acoustic evidence
                evidence_force = acoustic_evidence - traj.position
                force += evidence_force * confidence
                
                # Update momentum with step size
                traj.momentum += force * 0.01  
                
                # Clip momentum to prevent instability
                traj.momentum = np.clip(traj.momentum, -1.0, 1.0)
                
                # Update position
                traj.position += traj.momentum
            
            # Update history
            traj.history.append(traj.position.copy())
            
            # Update confidence
            self._update_confidence(traj, acoustic_evidence, confidence)
        
        # Prune low confidence trajectories
        self.prune_trajectories()
        
        # Ensure we don't exceed beam width
        if len(self.active_trajectories) > self.beam_search.beam_width:
            # Keep only the highest confidence trajectories
            active = sorted(
                self.active_trajectories,
                key=lambda t: t.confidence,
                reverse=True
            )[:self.beam_search.beam_width]
            
            # Mark others as pruned
            for traj in self.active_trajectories:
                if traj not in active:
                    traj.state = TrajectoryState.PRUNED
    
    def _create_trajectory(
        self,
        position: np.ndarray,
        confidence: float
    ) -> SemanticTrajectory:
        """
        Create a new trajectory
        
        Args:
            position: Initial position in semantic space
            confidence: Initial confidence score
            
        Returns:
            New trajectory instance
        """
        # Make a deep copy of the position vector
        position = np.array(position, copy=True)
        
        trajectory = SemanticTrajectory(
            id=self.next_trajectory_id,
            position=position,  # Use position directly without normalization
            momentum=np.zeros_like(position),
            confidence=confidence,
            state=TrajectoryState.ACTIVE,
            history=[position.copy()]
        )
        self.next_trajectory_id += 1
        return trajectory
    
    def _update_confidence(
        self,
        trajectory: SemanticTrajectory,
        evidence: np.ndarray,
        evidence_confidence: float
    ):
        """
        Update trajectory confidence based on new evidence
        
        Args:
            trajectory: Trajectory to update
            evidence: New evidence vector
            evidence_confidence: Confidence of the new evidence
        """
        # Compute similarity
        similarity = 1 - cosine(trajectory.position, evidence)
        
        # Update confidence with temporal decay
        trajectory.confidence = (
            0.7 * trajectory.confidence +
            0.3 * similarity * evidence_confidence
        )
        trajectory.confidence = np.clip(trajectory.confidence, 0.0, 1.0)
    
    def merge_similar_trajectories(self):
        """Merge trajectories that are close in semantic space"""
        active = self.active_trajectories
        if len(active) <= 1:
            return
            
        merged_ids = set()
        
        for i, t1 in enumerate(active):
            if t1.id in merged_ids:
                continue
                
            for t2 in active[i+1:]:
                if t2.id in merged_ids:
                    continue
                    
                # Compute similarity
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
        for trajectory in list(self.trajectories.values()):
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
        """Get the highest confidence trajectory"""
        active = self.active_trajectories
        if not active:
            return None
        return max(active, key=lambda t: t.confidence)
    
    def get_trajectory_paths(self) -> List[List[SemanticTrajectory]]:
        """Get all active trajectory paths"""
        active = self.active_trajectories
        if not active:
            return []
        # Sort by confidence and return each trajectory as its own path
        return [[t] for t in sorted(active, key=lambda t: t.confidence, reverse=True)]
