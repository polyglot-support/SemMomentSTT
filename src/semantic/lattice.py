"""
Lattice Module for SemMomentSTT

This module implements word lattice generation and processing:
- Lattice construction from beam search paths
- Node and edge management
- Path scoring and pruning
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple, NamedTuple
import numpy as np
from .types import SemanticTrajectory

class LatticeNode(NamedTuple):
    """Node in word lattice"""
    id: int
    word: str
    timestamp: float
    trajectory_id: int
    semantic_vector: np.ndarray
    confidence: float

class LatticeEdge(NamedTuple):
    """Edge in word lattice"""
    start_node: int  # Node ID
    end_node: int    # Node ID
    acoustic_score: float
    language_score: float
    semantic_score: float

@dataclass
class LatticePath:
    """Path through word lattice"""
    nodes: List[LatticeNode]
    edges: List[LatticeEdge]
    total_score: float
    acoustic_score: float
    language_score: float
    semantic_score: float

class WordLattice:
    def __init__(
        self,
        acoustic_weight: float = 0.4,
        language_weight: float = 0.3,
        semantic_weight: float = 0.3
    ):
        """
        Initialize word lattice
        
        Args:
            acoustic_weight: Weight for acoustic scores
            language_weight: Weight for language model scores
            semantic_weight: Weight for semantic similarity scores
        """
        self.acoustic_weight = acoustic_weight
        self.language_weight = language_weight
        self.semantic_weight = semantic_weight
        
        # Lattice structure
        self.nodes: Dict[int, LatticeNode] = {}
        self.edges: Dict[Tuple[int, int], LatticeEdge] = {}
        self.start_nodes: Set[int] = set()
        self.end_nodes: Set[int] = set()
        
        self.next_node_id = 0
    
    def add_node(
        self,
        word: str,
        timestamp: float,
        trajectory_id: int,
        semantic_vector: np.ndarray,
        confidence: float,
        is_start: bool = False,
        is_end: bool = False
    ) -> int:
        """
        Add node to lattice
        
        Args:
            word: Word at this node
            timestamp: Time of word
            trajectory_id: ID of associated trajectory
            semantic_vector: Semantic vector at node
            confidence: Node confidence
            is_start: Whether this is a start node
            is_end: Whether this is an end node
            
        Returns:
            ID of created node
        """
        node_id = self.next_node_id
        self.next_node_id += 1
        
        node = LatticeNode(
            id=node_id,
            word=word,
            timestamp=timestamp,
            trajectory_id=trajectory_id,
            semantic_vector=semantic_vector,
            confidence=confidence
        )
        
        self.nodes[node_id] = node
        
        if is_start:
            self.start_nodes.add(node_id)
        if is_end:
            self.end_nodes.add(node_id)
        
        return node_id
    
    def add_edge(
        self,
        start_node: int,
        end_node: int,
        acoustic_score: float,
        language_score: float,
        semantic_score: float
    ) -> None:
        """
        Add edge to lattice
        
        Args:
            start_node: ID of start node
            end_node: ID of end node
            acoustic_score: Acoustic model score
            language_score: Language model score
            semantic_score: Semantic similarity score
        """
        edge = LatticeEdge(
            start_node=start_node,
            end_node=end_node,
            acoustic_score=acoustic_score,
            language_score=language_score,
            semantic_score=semantic_score
        )
        
        self.edges[(start_node, end_node)] = edge
    
    def build_from_trajectories(
        self,
        trajectories: List[List[SemanticTrajectory]],
        word_scores: List[List[Tuple[str, float, float, float]]]  # (word, acoustic, lm, semantic)
    ) -> None:
        """
        Build lattice from trajectory paths
        
        Args:
            trajectories: List of trajectory paths
            word_scores: List of word scores for each path
        """
        # Clear existing lattice
        self.nodes.clear()
        self.edges.clear()
        self.start_nodes.clear()
        self.end_nodes.clear()
        
        # Process each path
        for path_idx, (traj_path, path_scores) in enumerate(zip(trajectories, word_scores)):
            prev_node_id = None
            
            for i, (trajectory, (word, acoustic, lm, semantic)) in enumerate(zip(traj_path, path_scores)):
                # Create node
                node_id = self.add_node(
                    word=word,
                    timestamp=trajectory.history[-1][0],  # Last timestamp
                    trajectory_id=trajectory.id,
                    semantic_vector=trajectory.position,
                    confidence=trajectory.confidence,
                    is_start=(i == 0),
                    is_end=(i == len(traj_path) - 1)
                )
                
                # Add edge from previous node
                if prev_node_id is not None:
                    self.add_edge(
                        start_node=prev_node_id,
                        end_node=node_id,
                        acoustic_score=acoustic,
                        language_score=lm,
                        semantic_score=semantic
                    )
                
                prev_node_id = node_id
    
    def find_best_paths(
        self,
        n_paths: int = 1,
        min_score: Optional[float] = None
    ) -> List[LatticePath]:
        """
        Find N best paths through lattice
        
        Args:
            n_paths: Number of paths to return
            min_score: Minimum path score threshold
            
        Returns:
            List of best paths
        """
        paths = []
        
        # Try each start node
        for start_node in self.start_nodes:
            paths.extend(self._find_paths_from_node(start_node))
        
        # Sort by score
        paths.sort(key=lambda p: p.total_score, reverse=True)
        
        # Filter by score threshold
        if min_score is not None:
            paths = [p for p in paths if p.total_score >= min_score]
        
        return paths[:n_paths]
    
    def _find_paths_from_node(
        self,
        start_node: int,
        visited: Optional[Set[int]] = None
    ) -> List[LatticePath]:
        """Find all paths from a start node"""
        if visited is None:
            visited = set()
        
        paths = []
        visited.add(start_node)
        
        # Base case: reached end node
        if start_node in self.end_nodes:
            return [LatticePath(
                nodes=[self.nodes[start_node]],
                edges=[],
                total_score=self.nodes[start_node].confidence,
                acoustic_score=0.0,
                language_score=0.0,
                semantic_score=0.0
            )]
        
        # Recursive case: try each outgoing edge
        for (node1, node2), edge in self.edges.items():
            if node1 != start_node:
                continue
            
            if node2 in visited:
                continue
            
            # Recursively find paths from next node
            sub_paths = self._find_paths_from_node(node2, visited.copy())
            
            # Add current node/edge to each sub-path
            for path in sub_paths:
                path.nodes.insert(0, self.nodes[start_node])
                path.edges.insert(0, edge)
                
                # Update scores
                path.acoustic_score += edge.acoustic_score * self.acoustic_weight
                path.language_score += edge.language_score * self.language_weight
                path.semantic_score += edge.semantic_score * self.semantic_weight
                
                path.total_score = (
                    path.acoustic_score +
                    path.language_score +
                    path.semantic_score
                ) / len(path.edges)
            
            paths.extend(sub_paths)
        
        return paths
    
    def prune(self, min_score: float) -> None:
        """
        Prune low-scoring nodes and edges
        
        Args:
            min_score: Minimum score threshold
        """
        # Find nodes to remove
        nodes_to_remove = set()
        for node_id, node in self.nodes.items():
            if node.confidence < min_score:
                nodes_to_remove.add(node_id)
        
        # Remove edges connected to pruned nodes
        edges_to_remove = set()
        for (start, end), edge in self.edges.items():
            if start in nodes_to_remove or end in nodes_to_remove:
                edges_to_remove.add((start, end))
            else:
                # Also prune low-scoring edges
                edge_score = (
                    edge.acoustic_score * self.acoustic_weight +
                    edge.language_score * self.language_weight +
                    edge.semantic_score * self.semantic_weight
                )
                if edge_score < min_score:
                    edges_to_remove.add((start, end))
        
        # Remove edges
        for edge_key in edges_to_remove:
            del self.edges[edge_key]
        
        # Remove nodes
        for node_id in nodes_to_remove:
            del self.nodes[node_id]
            self.start_nodes.discard(node_id)
            self.end_nodes.discard(node_id)
    
    def to_dot(self) -> str:
        """
        Convert lattice to DOT format for visualization
        
        Returns:
            DOT format string
        """
        dot = ["digraph {"]
        
        # Add nodes
        for node_id, node in self.nodes.items():
            # Color start/end nodes differently
            color = "lightblue" if node_id in self.start_nodes else (
                "lightgreen" if node_id in self.end_nodes else "white"
            )
            
            dot.append(
                f'  {node_id} [label="{node.word}\\n{node.confidence:.2f}" '
                f'style=filled fillcolor="{color}"];'
            )
        
        # Add edges
        for (start, end), edge in self.edges.items():
            # Combine scores for label
            score = (
                edge.acoustic_score * self.acoustic_weight +
                edge.language_score * self.language_weight +
                edge.semantic_score * self.semantic_weight
            )
            
            dot.append(
                f'  {start} -> {end} '
                f'[label="{score:.2f}"];'
            )
        
        dot.append("}")
        return "\n".join(dot)
