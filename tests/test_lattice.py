"""Tests for the lattice module"""

import pytest
import numpy as np
from src.semantic.lattice import (
    WordLattice,
    LatticeNode,
    LatticeEdge,
    LatticePath
)
from src.semantic.momentum_tracker import SemanticTrajectory, TrajectoryState

@pytest.fixture
def lattice():
    """Create a WordLattice instance for testing"""
    return WordLattice(
        acoustic_weight=0.4,
        language_weight=0.3,
        semantic_weight=0.3
    )

@pytest.fixture
def mock_trajectories():
    """Create mock trajectories for testing"""
    def create_trajectory(id: int, confidence: float) -> SemanticTrajectory:
        position = np.random.randn(768)
        position = position / np.linalg.norm(position)
        return SemanticTrajectory(
            id=id,
            position=position,
            momentum=np.zeros_like(position),
            confidence=confidence,
            state=TrajectoryState.ACTIVE,
            history=[(0.0, position)]  # Include timestamp
        )
    
    # Create two paths of trajectories
    path1 = [
        create_trajectory(1, 0.8),
        create_trajectory(2, 0.7),
        create_trajectory(3, 0.9)
    ]
    path2 = [
        create_trajectory(4, 0.6),
        create_trajectory(5, 0.8),
        create_trajectory(6, 0.7)
    ]
    
    return [path1, path2]

@pytest.fixture
def mock_word_scores():
    """Create mock word scores for testing"""
    path1_scores = [
        ("hello", 0.8, 0.7, 0.9),
        ("world", 0.7, 0.8, 0.6),
        ("test", 0.9, 0.8, 0.7)
    ]
    path2_scores = [
        ("hello", 0.6, 0.7, 0.8),
        ("there", 0.8, 0.7, 0.6),
        ("now", 0.7, 0.8, 0.9)
    ]
    return [path1_scores, path2_scores]

def test_lattice_initialization(lattice):
    """Test that WordLattice initializes correctly"""
    assert lattice.acoustic_weight == 0.4
    assert lattice.language_weight == 0.3
    assert lattice.semantic_weight == 0.3
    assert len(lattice.nodes) == 0
    assert len(lattice.edges) == 0
    assert len(lattice.start_nodes) == 0
    assert len(lattice.end_nodes) == 0

def test_node_creation(lattice):
    """Test node creation"""
    node_id = lattice.add_node(
        word="test",
        timestamp=0.5,
        trajectory_id=1,
        semantic_vector=np.random.randn(768),
        confidence=0.8,
        is_start=True,
        is_end=False
    )
    
    assert node_id in lattice.nodes
    assert node_id in lattice.start_nodes
    assert node_id not in lattice.end_nodes
    
    node = lattice.nodes[node_id]
    assert node.word == "test"
    assert node.timestamp == 0.5
    assert node.trajectory_id == 1
    assert node.confidence == 0.8

def test_edge_creation(lattice):
    """Test edge creation"""
    # Create two nodes
    node1 = lattice.add_node(
        word="hello",
        timestamp=0.0,
        trajectory_id=1,
        semantic_vector=np.random.randn(768),
        confidence=0.8
    )
    node2 = lattice.add_node(
        word="world",
        timestamp=0.5,
        trajectory_id=2,
        semantic_vector=np.random.randn(768),
        confidence=0.7
    )
    
    # Add edge
    lattice.add_edge(
        start_node=node1,
        end_node=node2,
        acoustic_score=0.8,
        language_score=0.7,
        semantic_score=0.9
    )
    
    assert (node1, node2) in lattice.edges
    edge = lattice.edges[(node1, node2)]
    assert edge.acoustic_score == 0.8
    assert edge.language_score == 0.7
    assert edge.semantic_score == 0.9

def test_lattice_construction(lattice, mock_trajectories, mock_word_scores):
    """Test lattice construction from trajectories"""
    lattice.build_from_trajectories(mock_trajectories, mock_word_scores)
    
    # Check nodes
    assert len(lattice.nodes) == 6  # Two paths of 3 nodes each
    assert len(lattice.start_nodes) == 2  # Two start nodes
    assert len(lattice.end_nodes) == 2  # Two end nodes
    
    # Check edges
    assert len(lattice.edges) == 4  # Two paths of 2 edges each

def test_path_finding(lattice, mock_trajectories, mock_word_scores):
    """Test finding best paths through lattice"""
    lattice.build_from_trajectories(mock_trajectories, mock_word_scores)
    
    # Find best paths
    paths = lattice.find_best_paths(n_paths=2)
    
    assert len(paths) <= 2
    assert all(isinstance(p, LatticePath) for p in paths)
    
    # Check path properties
    for path in paths:
        assert len(path.nodes) > 0
        assert len(path.edges) == len(path.nodes) - 1
        assert path.total_score > 0
        
        # Check score components
        assert path.acoustic_score >= 0
        assert path.language_score >= 0
        assert path.semantic_score >= 0

def test_lattice_pruning(lattice, mock_trajectories, mock_word_scores):
    """Test lattice pruning"""
    lattice.build_from_trajectories(mock_trajectories, mock_word_scores)
    
    initial_nodes = len(lattice.nodes)
    initial_edges = len(lattice.edges)
    
    # Prune with high threshold
    lattice.prune(min_score=0.8)
    
    assert len(lattice.nodes) < initial_nodes
    assert len(lattice.edges) < initial_edges

def test_dot_format(lattice, mock_trajectories, mock_word_scores):
    """Test DOT format generation"""
    lattice.build_from_trajectories(mock_trajectories, mock_word_scores)
    
    dot = lattice.to_dot()
    assert isinstance(dot, str)
    assert dot.startswith("digraph {")
    assert dot.endswith("}")
    
    # Check node and edge formatting
    for node_id, node in lattice.nodes.items():
        assert f"{node_id} [" in dot
        assert node.word in dot
    
    for (start, end) in lattice.edges:
        assert f"{start} -> {end}" in dot

def test_path_scores(lattice, mock_trajectories, mock_word_scores):
    """Test path score computation"""
    lattice.build_from_trajectories(mock_trajectories, mock_word_scores)
    paths = lattice.find_best_paths(n_paths=2)
    
    for path in paths:
        # Combined score should be weighted average
        expected_score = (
            path.acoustic_score +
            path.language_score +
            path.semantic_score
        ) / len(path.edges)
        
        assert abs(path.total_score - expected_score) < 1e-6

def test_empty_lattice(lattice):
    """Test operations on empty lattice"""
    paths = lattice.find_best_paths()
    assert len(paths) == 0
    
    lattice.prune(min_score=0.5)  # Should not raise error
    
    dot = lattice.to_dot()
    assert "digraph {" in dot

def test_disconnected_nodes(lattice):
    """Test handling of disconnected nodes"""
    # Add nodes without edges
    node1 = lattice.add_node(
        word="test1",
        timestamp=0.0,
        trajectory_id=1,
        semantic_vector=np.random.randn(768),
        confidence=0.8,
        is_start=True
    )
    node2 = lattice.add_node(
        word="test2",
        timestamp=0.5,
        trajectory_id=2,
        semantic_vector=np.random.randn(768),
        confidence=0.7,
        is_end=True
    )
    
    paths = lattice.find_best_paths()
    assert len(paths) == 0  # No valid paths without edges
