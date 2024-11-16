"""
Common types for semantic processing modules
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set
import numpy as np

class TrajectoryState(Enum):
    """State of a semantic trajectory"""
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

@dataclass
class BeamHypothesis:
    """Container for a beam search hypothesis"""
    trajectory: SemanticTrajectory
    score: float
    parent_id: Optional[int]  # ID of parent trajectory
    children_ids: Set[int]    # IDs of child trajectories
    depth: int               # Depth in the beam tree

@dataclass
class WordScore:
    """Word-level scoring information"""
    word: str
    confidence: float
    semantic_similarity: float
    language_model_score: float
    start_time: float
    duration: float

@dataclass
class NBestHypothesis:
    """N-best hypothesis with detailed scoring"""
    text: str
    confidence: float
    word_scores: List[WordScore]
    trajectory_path: List[SemanticTrajectory]

@dataclass
class LatticeNode:
    """Node in word lattice"""
    id: int
    word: str
    start_time: float
    duration: float
    score: float

@dataclass
class LatticeEdge:
    """Edge in word lattice"""
    source_id: int
    target_id: int
    score: float

@dataclass
class LatticePath:
    """Complete path through word lattice"""
    nodes: List[LatticeNode]
    edges: List[LatticeEdge]
    total_score: float
