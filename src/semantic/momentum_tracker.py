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
        momentum_decay: float = 0.98,  # Increased for more stable trajectories
        min_confidence: float = 0.05,  # Lowered to be less strict
        merge_threshold: float = 0.75,  # Lowered to allow more merging
        force_scale: float = 0.5,  # Reduced for more stable movement
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
            diversity_penalty=0.2  # Increased for more diverse paths
        )
        
        self.trajectories: Dict[int, SemanticTrajectory] = {}
        self.next_trajectory_id = 0
        
        # Initialize attraction centers in a more structured way
        # Create evenly spaced centers
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        centers = []
        for angle in angles:
            center = np.zeros(semantic_dim)
            center[0] = np.cos(angle)
            center[1] = np.sin(angle)
            centers.append(center)
        self.attraction_centers = np.array(centers)
        self.attraction_centers = self.attraction_centers / np.linalg.norm(
            self.attraction_centers, axis=1, keepdims=True
        )
        self.attraction_strengths = np.ones(8) * 0.02  # Reduced strength
    
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
        
        # Add forces from attraction centers with quadratic falloff
        for center, strength in zip(self.attraction_centers, self.attraction_strengths):
            direction = center - position
            distance = max(np.linalg.norm(direction), 1e-6)
            force += strength * direction / (distance * distance)
        
        # Add repulsive forces from other trajectories with cubic falloff
        for trajectory in self.active_trajectories:
            if not np.array_equal(trajectory.position, position):
                direction = trajectory.position - position
                distance = max(np.linalg.norm(direction), 1e-6)
                force -= 0.005 * direction / (distance * distance * distance)  # Reduced repulsion
        
        # Scale force and clip to prevent extreme values
        force *= self.force_scale
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > 1.0:
            force = force / force_magnitude
        
        return force
    
    def _update_momentum(self, trajectory: SemanticTrajectory, force: np.ndarray):
        """Update trajectory momentum with decay and new force"""
        # Apply momentum decay while preserving direction
        if not np.allclose(trajectory.momentum, 0):
            momentum_magnitude = np.linalg.norm(trajectory.momentum)
            if momentum_magnitude > 1e-6:
                momentum_dir = trajectory.momentum / momentum_magnitude
                new_magnitude = momentum_magnitude * self.momentum_decay
                trajectory.momentum = momentum_dir * new_magnitude
        
        # Add force contribution with smoothing
        if not np.allclose(force, 0):
            force_magnitude = np.linalg.norm(force)
            if force_magnitude > 1e-6:
                force_dir = force / force_magnitude
                # Smooth force addition
                if np.allclose(trajectory.momentum, 0):
                    trajectory.momentum = force_dir * self.force_scale * 0.25
                else:
                    # Blend new force with current momentum direction
                    current_dir = trajectory.momentum / np.linalg.norm(trajectory.momentum)
                    blended_dir = 0.7 * current_dir + 0.3 * force_dir
                    blended_dir = blended_dir / np.linalg.norm(blended_dir)
                    trajectory.momentum += blended_dir * self.force_scale * 0.25
    
    def update_trajectories(self, acoustic_evidence: np.ndarray, confidence: float):
        """
        Update all active trajectories based on new evidence
        
        Args:
            acoustic_evidence: New acoustic evidence vector
            confidence: Confidence score for the new evidence
        """
        # Make a copy of the evidence vector
        acoustic_evidence = np.array(acoustic_evidence, copy=True)
        acoustic_evidence = acoustic_evidence / np.linalg.norm(acoustic_evidence)
        
        # Try to merge similar trajectories first
        self.merge_similar_trajectories()
        
        # Check if we can merge with any existing trajectory
        merged = False
        if self.active_trajectories:
            for traj in self.active_trajectories:
                similarity = 1 - cosine(traj.position, acoustic_evidence)
                if similarity > self.merge_threshold:
                    # Update confidence before merging
                    new_confidence = max(traj.confidence, confidence)
                    
                    # Merge positions with proper normalization
                    weight1 = traj.confidence / (traj.confidence + confidence)
                    weight2 = confidence / (traj.confidence + confidence)
                    merged_pos = weight1 * traj.position + weight2 * acoustic_evidence
                    pos_norm = np.linalg.norm(merged_pos)
                    if pos_norm > 1e-6:
                        traj.position = merged_pos / pos_norm
                    
                    # Update momentum with smooth transition
                    if not np.allclose(traj.momentum, 0):
                        momentum_dir = traj.momentum / np.linalg.norm(traj.momentum)
                        traj.momentum = momentum_dir * np.linalg.norm(traj.momentum) * self.momentum_decay
                    
                    traj.confidence = new_confidence
                    merged = True
                    break
        
        # Create new trajectory if not merged
        if not merged and confidence >= self.min_confidence and len(self.active_trajectories) < self.beam_search.beam_width:
            trajectory = self._create_trajectory(acoustic_evidence, confidence)
            self.trajectories[trajectory.id] = trajectory
        
        # Update existing trajectories
        for traj in self.active_trajectories:
            # First update - apply force to build momentum
            if np.allclose(traj.momentum, 0):
                force = acoustic_evidence - traj.position
                force_magnitude = np.linalg.norm(force)
                if force_magnitude > 1e-6:
                    force_dir = force / force_magnitude
                    traj.momentum = force_dir * self.force_scale * 0.25
            else:
                # Compute and apply forces
                force = self.compute_force_field(traj.position)
                evidence_force = acoustic_evidence - traj.position
                force += evidence_force * confidence * 0.5  # Reduced evidence influence
                
                # Update momentum with decay and new force
                self._update_momentum(traj, force)
            
            # Update position with momentum clamping
            if not np.allclose(traj.momentum, 0):
                # Clamp momentum magnitude
                momentum_magnitude = np.linalg.norm(traj.momentum)
                if momentum_magnitude > 0.1:  # Maximum step size
                    traj.momentum = (traj.momentum / momentum_magnitude) * 0.1
                
                traj.position += traj.momentum
                pos_norm = np.linalg.norm(traj.position)
                if pos_norm > 1e-6:
                    traj.position = traj.position / pos_norm
            
            # Update history
            traj.history.append(traj.position.copy())
            
            # Update confidence with temporal smoothing
            self._update_confidence(traj, acoustic_evidence, confidence)
        
        # Try to merge similar trajectories again
        self.merge_similar_trajectories()
        
        # Prune low confidence trajectories
        self.prune_trajectories()
        
        # Ensure we don't exceed beam width
        if len(self.active_trajectories) > self.beam_search.beam_width:
            active = sorted(
                self.active_trajectories,
                key=lambda t: t.confidence,
                reverse=True
            )[:self.beam_search.beam_width]
            
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
        # Make a deep copy and normalize the position vector
        position = np.array(position, copy=True)
        position = position / np.linalg.norm(position)
        
        # Set initial state based on confidence
        state = TrajectoryState.ACTIVE if confidence >= self.min_confidence else TrajectoryState.PRUNED
        
        trajectory = SemanticTrajectory(
            id=self.next_trajectory_id,
            position=position,
            momentum=np.zeros_like(position),
            confidence=confidence,
            state=state,
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
        # Compute similarity with cosine distance
        similarity = 1 - cosine(trajectory.position, evidence)
        
        # Update confidence with temporal smoothing
        trajectory.confidence = (
            0.8 * trajectory.confidence +  # Increased history weight
            0.2 * similarity * evidence_confidence  # Reduced new evidence weight
        )
        trajectory.confidence = np.clip(trajectory.confidence, 0.0, 1.0)
        
        # Mark as pruned if below threshold
        if trajectory.confidence < self.min_confidence:
            trajectory.state = TrajectoryState.PRUNED
    
    def merge_similar_trajectories(self):
        """Merge trajectories that are close in semantic space"""
        active = self.active_trajectories
        if len(active) <= 1:
            return
        
        # Sort by confidence to preserve highest confidence trajectories
        active = sorted(active, key=lambda t: t.confidence, reverse=True)
        
        for i, t1 in enumerate(active):
            for t2 in active[i+1:]:
                if t2.state != TrajectoryState.ACTIVE:
                    continue
                    
                similarity = 1 - cosine(t1.position, t2.position)
                
                if similarity > self.merge_threshold:
                    # Update confidence before merging
                    new_confidence = max(t1.confidence, t2.confidence)
                    
                    # Merge positions with proper normalization
                    weight1 = t1.confidence / (t1.confidence + t2.confidence)
                    weight2 = t2.confidence / (t1.confidence + t2.confidence)
                    merged_pos = weight1 * t1.position + weight2 * t2.position
                    pos_norm = np.linalg.norm(merged_pos)
                    if pos_norm > 1e-6:
                        t1.position = merged_pos / pos_norm
                    
                    # Update momentum with smooth blending
                    if not np.allclose(t1.momentum, 0) and not np.allclose(t2.momentum, 0):
                        m1_norm = np.linalg.norm(t1.momentum)
                        m2_norm = np.linalg.norm(t2.momentum)
                        if m1_norm > 1e-6:
                            # Blend momentum directions
                            m1_dir = t1.momentum / m1_norm
                            m2_dir = t2.momentum / m2_norm
                            merged_dir = weight1 * m1_dir + weight2 * m2_dir
                            merged_dir = merged_dir / np.linalg.norm(merged_dir)
                            # Use weighted average of magnitudes
                            merged_mag = weight1 * m1_norm + weight2 * m2_norm
                            t1.momentum = merged_dir * merged_mag
                    
                    t1.confidence = new_confidence
                    t2.state = TrajectoryState.MERGED
                    return  # Exit after successful merge
    
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
        return [[t] for t in sorted(active, key=lambda t: t.confidence, reverse=True)]
