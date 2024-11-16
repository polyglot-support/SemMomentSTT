"""
Beam Search Module for SemMomentSTT

This module implements beam search for managing multiple trajectory hypotheses:
- Beam scoring and ranking
- Hypothesis pruning
- Path reconstruction
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
from .momentum_tracker import SemanticTrajectory, TrajectoryState

@dataclass
class BeamHypothesis:
    """Container for a beam search hypothesis"""
    trajectory: SemanticTrajectory
    score: float
    parent_id: Optional[int]  # ID of parent trajectory
    children_ids: Set[int]    # IDs of child trajectories
    depth: int               # Depth in the beam tree

class BeamSearch:
    def __init__(
        self,
        beam_width: int = 5,
        max_depth: int = 10,
        score_threshold: float = 0.1,
        diversity_penalty: float = 0.1
    ):
        """
        Initialize beam search
        
        Args:
            beam_width: Maximum number of active beams
            max_depth: Maximum depth of beam tree
            score_threshold: Minimum score for keeping a beam
            diversity_penalty: Penalty for similar trajectories
        """
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.score_threshold = score_threshold
        self.diversity_penalty = diversity_penalty
        
        # Beam management
        self.hypotheses: Dict[int, BeamHypothesis] = {}
        self.active_beams: List[BeamHypothesis] = []
        self.completed_beams: List[BeamHypothesis] = []
    
    def score_hypothesis(
        self,
        trajectory: SemanticTrajectory,
        parent: Optional[BeamHypothesis] = None
    ) -> float:
        """
        Compute score for a trajectory hypothesis
        
        Args:
            trajectory: Trajectory to score
            parent: Optional parent hypothesis
            
        Returns:
            Combined score for the hypothesis
        """
        # Base score from trajectory confidence
        score = trajectory.confidence
        
        if parent is not None:
            # Add momentum consistency with parent
            momentum_diff = np.linalg.norm(
                trajectory.momentum - parent.trajectory.momentum
            )
            momentum_score = np.exp(-momentum_diff)
            score *= 0.7 * momentum_score
            
            # Add semantic consistency with parent
            semantic_diff = np.linalg.norm(
                trajectory.position - parent.trajectory.position
            )
            semantic_score = np.exp(-semantic_diff)
            score *= 0.3 * semantic_score
        
        # Apply diversity penalty
        if self.active_beams:
            similarities = [
                self._compute_similarity(trajectory, beam.trajectory)
                for beam in self.active_beams
            ]
            max_similarity = max(similarities)
            score -= self.diversity_penalty * max_similarity
        
        return score
    
    def _compute_similarity(
        self,
        traj1: SemanticTrajectory,
        traj2: SemanticTrajectory
    ) -> float:
        """Compute similarity between two trajectories"""
        # Position similarity
        pos_sim = np.dot(traj1.position, traj2.position) / (
            np.linalg.norm(traj1.position) * np.linalg.norm(traj2.position)
        )
        
        # Momentum similarity
        mom_sim = np.dot(traj1.momentum, traj2.momentum) / (
            np.linalg.norm(traj1.momentum) * np.linalg.norm(traj2.momentum)
            + 1e-6  # Avoid division by zero
        )
        
        return 0.7 * pos_sim + 0.3 * mom_sim
    
    def update_beams(
        self,
        trajectories: List[SemanticTrajectory]
    ) -> List[BeamHypothesis]:
        """
        Update beam hypotheses with new trajectories
        
        Args:
            trajectories: List of new trajectories to consider
            
        Returns:
            List of current best hypotheses
        """
        # Score all combinations of new trajectories with existing beams
        new_hypotheses: List[BeamHypothesis] = []
        
        # Handle initial case
        if not self.active_beams:
            for traj in trajectories:
                score = self.score_hypothesis(traj)
                if score >= self.score_threshold:
                    hypothesis = BeamHypothesis(
                        trajectory=traj,
                        score=score,
                        parent_id=None,
                        children_ids=set(),
                        depth=0
                    )
                    new_hypotheses.append(hypothesis)
        else:
            # Extend existing beams
            for parent in self.active_beams:
                if parent.depth >= self.max_depth:
                    continue
                    
                for traj in trajectories:
                    score = self.score_hypothesis(traj, parent)
                    if score >= self.score_threshold:
                        hypothesis = BeamHypothesis(
                            trajectory=traj,
                            score=score,
                            parent_id=parent.trajectory.id,
                            children_ids=set(),
                            depth=parent.depth + 1
                        )
                        new_hypotheses.append(hypothesis)
                        parent.children_ids.add(traj.id)
        
        # Select top-k beams
        all_beams = self.active_beams + new_hypotheses
        sorted_beams = sorted(
            all_beams,
            key=lambda h: h.score,
            reverse=True
        )
        
        # Update active beams
        self.active_beams = sorted_beams[:self.beam_width]
        
        # Move completed beams
        for beam in sorted_beams[self.beam_width:]:
            if not beam.children_ids:  # No children, consider completed
                self.completed_beams.append(beam)
        
        return self.active_beams
    
    def get_best_path(self) -> List[SemanticTrajectory]:
        """
        Get the highest scoring path through the beam tree
        
        Returns:
            List of trajectories forming the best path
        """
        if not self.active_beams and not self.completed_beams:
            return []
        
        # Find best final beam
        all_beams = self.active_beams + self.completed_beams
        best_beam = max(all_beams, key=lambda h: h.score)
        
        # Reconstruct path
        path = [best_beam.trajectory]
        current_id = best_beam.parent_id
        
        while current_id is not None:
            # Find parent hypothesis
            for beam in all_beams:
                if beam.trajectory.id == current_id:
                    path.append(beam.trajectory)
                    current_id = beam.parent_id
                    break
        
        return list(reversed(path))  # Return in chronological order
    
    def prune_beams(self, min_score: Optional[float] = None):
        """
        Prune low-scoring beams
        
        Args:
            min_score: Optional minimum score threshold
        """
        threshold = min_score or self.score_threshold
        
        # Prune active beams
        self.active_beams = [
            beam for beam in self.active_beams
            if beam.score >= threshold
        ]
        
        # Prune completed beams
        self.completed_beams = [
            beam for beam in self.completed_beams
            if beam.score >= threshold
        ]
    
    def reset(self):
        """Reset beam search state"""
        self.hypotheses.clear()
        self.active_beams.clear()
        self.completed_beams.clear()
