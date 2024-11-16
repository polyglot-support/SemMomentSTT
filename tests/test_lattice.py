"""Tests for the lattice module"""

import pytest
import numpy as np
from src.semantic.lattice import (
    WordLattice,
    LatticeNode,
    LatticeEdge,
    LatticePath
)
from src.semantic.types import SemanticTrajectory, TrajectoryState

class TestWordLattice:
    """Test suite for WordLattice class"""

    @pytest.fixture
    def lattice(self):
        """Create a WordLattice instance for testing"""
        return WordLattice(
            acoustic_weight=0.4,
            language_weight=0.3,
            semantic_weight=0.3
        )

    class TestInitialization:
        """Tests for WordLattice initialization"""

        def test_weight_settings(self, lattice):
            """Test weight configurations"""
            assert lattice.acoustic_weight == 0.4
            assert lattice.language_weight == 0.3
            assert lattice.semantic_weight == 0.3

        def test_initial_collections(self, lattice):
            """Test initial collection states"""
            assert len(lattice.nodes) == 0
            assert len(lattice.edges) == 0
            assert len(lattice.start_nodes) == 0
            assert len(lattice.end_nodes) == 0

    class TestNodeOperations:
        """Tests for node operations"""

        def test_basic_node_creation(self, lattice, mock_vector):
            """Test basic node creation"""
            node_id = lattice.add_node(
                word="test",
                timestamp=0.5,
                trajectory_id=1,
                semantic_vector=mock_vector,
                confidence=0.8
            )
            assert node_id in lattice.nodes
            node = lattice.nodes[node_id]
            assert node.word == "test"
            assert node.timestamp == 0.5
            assert node.trajectory_id == 1
            assert node.confidence == 0.8

        def test_start_node_creation(self, lattice, mock_vector):
            """Test start node creation"""
            node_id = lattice.add_node(
                word="start",
                timestamp=0.0,
                trajectory_id=1,
                semantic_vector=mock_vector,
                confidence=0.8,
                is_start=True
            )
            assert node_id in lattice.start_nodes

        def test_end_node_creation(self, lattice, mock_vector):
            """Test end node creation"""
            node_id = lattice.add_node(
                word="end",
                timestamp=1.0,
                trajectory_id=1,
                semantic_vector=mock_vector,
                confidence=0.8,
                is_end=True
            )
            assert node_id in lattice.end_nodes

    class TestEdgeOperations:
        """Tests for edge operations"""

        def test_basic_edge_creation(self, lattice, mock_vector):
            """Test basic edge creation"""
            node1 = lattice.add_node(
                word="hello",
                timestamp=0.0,
                trajectory_id=1,
                semantic_vector=mock_vector,
                confidence=0.8
            )
            node2 = lattice.add_node(
                word="world",
                timestamp=0.5,
                trajectory_id=2,
                semantic_vector=mock_vector,
                confidence=0.7
            )
            
            lattice.add_edge(
                start_node=node1,
                end_node=node2,
                acoustic_score=0.8,
                language_score=0.7,
                semantic_score=0.9
            )
            
            assert (node1, node2) in lattice.edges

        def test_edge_scores(self, lattice, mock_vector):
            """Test edge score properties"""
            node1 = lattice.add_node(
                word="test1",
                timestamp=0.0,
                trajectory_id=1,
                semantic_vector=mock_vector,
                confidence=0.8
            )
            node2 = lattice.add_node(
                word="test2",
                timestamp=0.5,
                trajectory_id=2,
                semantic_vector=mock_vector,
                confidence=0.7
            )
            
            lattice.add_edge(
                start_node=node1,
                end_node=node2,
                acoustic_score=0.8,
                language_score=0.7,
                semantic_score=0.9
            )
            
            edge = lattice.edges[(node1, node2)]
            assert edge.acoustic_score == 0.8
            assert edge.language_score == 0.7
            assert edge.semantic_score == 0.9

    class TestLatticeConstruction:
        """Tests for lattice construction"""

        def test_trajectory_based_construction(self, lattice, mock_trajectories, mock_word_scores):
            """Test lattice construction from trajectories"""
            lattice.build_from_trajectories(mock_trajectories, [mock_word_scores])
            
            assert len(lattice.nodes) == 3  # Three nodes per path
            assert len(lattice.start_nodes) == 1
            assert len(lattice.end_nodes) == 1
            assert len(lattice.edges) == 2

        def test_node_connectivity(self, lattice, mock_trajectories, mock_word_scores):
            """Test node connectivity after construction"""
            lattice.build_from_trajectories(mock_trajectories, [mock_word_scores])
            
            # Check that nodes are properly connected
            for (start, end) in lattice.edges:
                assert start in lattice.nodes
                assert end in lattice.nodes
                assert lattice.nodes[end].timestamp > lattice.nodes[start].timestamp

    class TestPathFinding:
        """Tests for path finding functionality"""

        def test_best_paths_structure(self, lattice, mock_trajectories, mock_word_scores):
            """Test structure of best paths"""
            lattice.build_from_trajectories(mock_trajectories, [mock_word_scores])
            paths = lattice.find_best_paths(n_paths=2)
            
            assert len(paths) <= 2
            assert all(isinstance(p, LatticePath) for p in paths)

        def test_path_properties(self, lattice, mock_trajectories, mock_word_scores):
            """Test properties of found paths"""
            lattice.build_from_trajectories(mock_trajectories, [mock_word_scores])
            paths = lattice.find_best_paths(n_paths=2)
            
            for path in paths:
                assert len(path.nodes) > 0
                assert len(path.edges) == len(path.nodes) - 1
                assert path.total_score > 0

        def test_path_scores(self, lattice, mock_trajectories, mock_word_scores):
            """Test path score computation"""
            lattice.build_from_trajectories(mock_trajectories, [mock_word_scores])
            paths = lattice.find_best_paths(n_paths=2)
            
            for path in paths:
                expected_score = (
                    path.acoustic_score +
                    path.language_score +
                    path.semantic_score
                ) / len(path.edges)
                assert abs(path.total_score - expected_score) < 1e-6

    class TestLatticeOperations:
        """Tests for lattice operations"""

        def test_pruning(self, lattice, mock_trajectories, mock_word_scores):
            """Test lattice pruning"""
            lattice.build_from_trajectories(mock_trajectories, [mock_word_scores])
            
            initial_nodes = len(lattice.nodes)
            initial_edges = len(lattice.edges)
            
            lattice.prune(min_score=0.8)
            
            assert len(lattice.nodes) < initial_nodes
            assert len(lattice.edges) < initial_edges

        def test_dot_format_generation(self, lattice, mock_trajectories, mock_word_scores):
            """Test DOT format generation"""
            lattice.build_from_trajectories(mock_trajectories, [mock_word_scores])
            
            dot = lattice.to_dot()
            assert isinstance(dot, str)
            assert dot.startswith("digraph {")
            assert dot.endswith("}")

        def test_dot_format_content(self, lattice, mock_trajectories, mock_word_scores):
            """Test DOT format content"""
            lattice.build_from_trajectories(mock_trajectories, [mock_word_scores])
            dot = lattice.to_dot()
            
            for node_id, node in lattice.nodes.items():
                assert f"{node_id} [" in dot
                assert node.word in dot
            
            for (start, end) in lattice.edges:
                assert f"{start} -> {end}" in dot

    class TestEdgeCases:
        """Tests for edge cases"""

        def test_empty_lattice_operations(self, lattice):
            """Test operations on empty lattice"""
            paths = lattice.find_best_paths()
            assert len(paths) == 0
            
            lattice.prune(min_score=0.5)
            assert len(lattice.nodes) == 0
            assert len(lattice.edges) == 0
            
            dot = lattice.to_dot()
            assert "digraph {" in dot

        def test_disconnected_nodes(self, lattice, mock_vector):
            """Test handling of disconnected nodes"""
            node1 = lattice.add_node(
                word="test1",
                timestamp=0.0,
                trajectory_id=1,
                semantic_vector=mock_vector,
                confidence=0.8,
                is_start=True
            )
            node2 = lattice.add_node(
                word="test2",
                timestamp=0.5,
                trajectory_id=2,
                semantic_vector=mock_vector,
                confidence=0.7,
                is_end=True
            )
            
            paths = lattice.find_best_paths()
            assert len(paths) == 0
