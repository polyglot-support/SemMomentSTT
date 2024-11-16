"""
Semantic Momentum Tracker for SemMomentSTT

This module implements the core semantic momentum tracking functionality:
- Multi-trajectory state maintenance
- Momentum vector computation
- Semantic force field modeling
- Beam search for trajectory management
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
from scipy.spatial.distance import cosine
import torch
import torch.nn.functional as F
from .beam_search import BeamSearch, BeamHypothesis

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
        force_scale: float = 1.0,
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
        # Create new trajectory
        trajectory = self._create_trajectory(acoustic_evidence, confidence)
        
        # Update existing trajectories with forces
        for traj in self.active_trajectories:
            # Apply semantic forces
            force = self.compute_force_field(traj.position)
            
            # Add force from acoustic evidence
            evidence_force = acoustic_evidence - traj.position
            force += evidence_force * confidence
            
            # Update trajectory
            traj.update_position(force)
            
            # Apply momentum decay
            traj.momentum *= self.momentum_decay
            
            # Update confidence based on consistency and evidence
            self._update_confidence(traj, acoustic_evidence, confidence)
        
        # Update beam search with all trajectories
        beams = self.beam_search.update_beams(self.active_trajectories)
        
        # Update trajectory states based on beam search
        active_ids = {beam.trajectory.id for beam in beams}
        for traj in self.active_trajectories:
            if traj.id not in active_ids:
                traj.state = TrajectoryState.PRUNED
    
    def _create_trajectory(
        self,
        position: np.ndarray,
        confidence: float
    ) -> SemanticTrajectory:
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
        return trajectory
    
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
        # Use beam search pruning
        self.beam_search.prune_beams(self.min_confidence)
        
        # Update trajectory states
        active_ids = {
            beam.trajectory.id
            for beam in self.beam_search.active_beams
        }
        for trajectory in self.active_trajectories:
            if trajectory.id not in active_ids:
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
        # Get best path from beam search
        path = self.beam_search.get_best_path()
        if path:
            return path[-1]  # Return most recent trajectory
        return None
    
    def get_trajectory_paths(self) -> List[List[SemanticTrajectory]]:
        """
        Get all active trajectory paths from beam search
        
        Returns:
            List of trajectory paths, ordered by score
        """
        paths = []
        for beam in self.beam_search.active_beams:
            path = []
            current = beam
            while current is not None:
                path.append(current.trajectory)
                parent_id = current.parent_id
                current = next(
                    (b for b in self.beam_search.active_beams
                     if b.trajectory.id == parent_id),
                    None
                )
            paths.append(list(reversed(path)))
        return sorted(
            paths,
            key=lambda p: p[-1].confidence,
            reverse=True
        )
