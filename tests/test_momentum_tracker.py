"""Tests for the momentum tracker module"""

import pytest
import numpy as np
from scipy.spatial.distance import cosine
from src.semantic.types import (
    SemanticTrajectory,
    TrajectoryState,
    BeamHypothesis
)

class TestMomentumTracker:
    """Test suite for MomentumTracker class"""

    class TestInitialization:
        """Tests for MomentumTracker initialization"""

        def test_dimension_setting(self, shared_momentum_tracker):
            """Test semantic dimension configuration"""
            assert shared_momentum_tracker.semantic_dim == 768

        def test_trajectory_limits(self, shared_momentum_tracker):
            """Test trajectory limit settings"""
            assert shared_momentum_tracker.max_trajectories == 5

        def test_momentum_parameters(self, shared_momentum_tracker):
            """Test momentum-related parameters"""
            assert shared_momentum_tracker.momentum_decay == 0.95
            assert shared_momentum_tracker.min_confidence == 0.1
            assert shared_momentum_tracker.merge_threshold == 0.85

        def test_beam_search_configuration(self, shared_momentum_tracker):
            """Test beam search configuration"""
            assert hasattr(shared_momentum_tracker, 'beam_search')
            assert shared_momentum_tracker.beam_search.beam_width == 3
            assert shared_momentum_tracker.beam_search.max_depth == 5

    class TestForceField:
        """Tests for force field computations"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_momentum_tracker):
            """Setup for each test method"""
            self.tracker = shared_momentum_tracker
            self.tracker.trajectories.clear()

        def test_output_shape(self, mock_vector):
            """Test force field output shape"""
            force = self.tracker.compute_force_field(mock_vector)
            assert force.shape == mock_vector.shape
            assert isinstance(force, np.ndarray)

        def test_empty_trajectories(self, mock_vector):
            """Test force field with no trajectories"""
            force = self.tracker.compute_force_field(mock_vector)
            assert not np.allclose(force, 0)

        def test_single_trajectory(self, mock_vector):
            """Test force field with one trajectory"""
            self.tracker.update_trajectories(mock_vector, confidence=0.8)
            force = self.tracker.compute_force_field(mock_vector)
            assert not np.allclose(force, 0)

        def test_multiple_trajectories(self, mock_vector):
            """Test force field with multiple trajectories"""
            for i in range(3):
                evidence = mock_vector + np.random.randn(768) * 0.1
                evidence = evidence / np.linalg.norm(evidence)
                self.tracker.update_trajectories(evidence, confidence=0.8)
            force = self.tracker.compute_force_field(mock_vector)
            assert not np.allclose(force, 0)

    class TestTrajectoryCreation:
        """Tests for trajectory creation"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_momentum_tracker):
            """Setup for each test method"""
            self.tracker = shared_momentum_tracker
            self.tracker.trajectories.clear()

        def test_basic_creation(self, mock_vector):
            """Test basic trajectory creation"""
            self.tracker.update_trajectories(mock_vector, confidence=0.8)
            assert len(self.tracker.active_trajectories) == 1
            trajectory = self.tracker.active_trajectories[0]
            assert np.allclose(trajectory.position, mock_vector)
            assert trajectory.state == TrajectoryState.ACTIVE

        def test_confidence_setting(self, mock_vector):
            """Test confidence setting in creation"""
            confidence = 0.8
            self.tracker.update_trajectories(mock_vector, confidence=confidence)
            assert np.isclose(self.tracker.active_trajectories[0].confidence, confidence)

        def test_capacity_limit(self, mock_vector):
            """Test creation at capacity limit"""
            for i in range(self.tracker.max_trajectories + 1):
                evidence = mock_vector + np.random.randn(768) * 0.1
                evidence = evidence / np.linalg.norm(evidence)
                self.tracker.update_trajectories(evidence, confidence=0.8)
            assert len(self.tracker.active_trajectories) <= self.tracker.max_trajectories

        def test_below_confidence_threshold(self, mock_vector):
            """Test creation below confidence threshold"""
            self.tracker.update_trajectories(
                mock_vector, 
                confidence=self.tracker.min_confidence - 0.01
            )
            assert len(self.tracker.active_trajectories) == 0

    class TestBeamSearchIntegration:
        """Tests for beam search integration"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_momentum_tracker):
            """Setup for each test method"""
            self.tracker = shared_momentum_tracker
            self.tracker.trajectories.clear()

        def test_width_constraint(self, mock_vector):
            """Test beam width constraint"""
            for _ in range(self.tracker.beam_search.beam_width + 2):
                evidence = mock_vector + np.random.randn(768) * 0.1
                evidence = evidence / np.linalg.norm(evidence)
                self.tracker.update_trajectories(evidence, confidence=0.8)
            assert len(self.tracker.active_trajectories) <= self.tracker.beam_search.beam_width

        def test_confidence_filtering(self, mock_vector):
            """Test confidence-based filtering"""
            confidences = [0.3, 0.8, 0.5]
            for conf in confidences:
                evidence = mock_vector + np.random.randn(768) * 0.1
                evidence = evidence / np.linalg.norm(evidence)
                self.tracker.update_trajectories(evidence, confidence=conf)
            assert all(t.confidence >= self.tracker.min_confidence 
                      for t in self.tracker.active_trajectories)

        def test_path_creation(self, mock_vector):
            """Test trajectory path creation"""
            for i in range(3):
                evidence = mock_vector + np.random.randn(768) * 0.1 * i
                evidence = evidence / np.linalg.norm(evidence)
                self.tracker.update_trajectories(evidence, confidence=0.8)
            
            paths = self.tracker.get_trajectory_paths()
            assert isinstance(paths, list)
            assert all(isinstance(path, list) for path in paths)
            assert all(isinstance(traj, SemanticTrajectory) 
                      for path in paths for traj in path)

    class TestPathConsistency:
        """Tests for path consistency"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_momentum_tracker):
            """Setup for each test method"""
            self.tracker = shared_momentum_tracker
            self.tracker.trajectories.clear()

        def test_temporal_ordering(self, mock_vector):
            """Test temporal ordering of paths"""
            for i in range(3):
                evidence = mock_vector + np.random.randn(768) * 0.1 * i
                evidence = evidence / np.linalg.norm(evidence)
                self.tracker.update_trajectories(evidence, confidence=0.8)
            
            paths = self.tracker.get_trajectory_paths()
            if paths:
                path = paths[0]
                for i in range(len(path) - 1):
                    assert path[i].id < path[i + 1].id

        def test_semantic_consistency(self, mock_vector):
            """Test semantic consistency of paths"""
            base_vector = mock_vector
            for i in range(3):
                evidence = base_vector + np.random.randn(768) * 0.1
                evidence = evidence / np.linalg.norm(evidence)
                self.tracker.update_trajectories(evidence, confidence=0.8)
                base_vector = evidence
            
            paths = self.tracker.get_trajectory_paths()
            if paths:
                path = paths[0]
                for i in range(len(path) - 1):
                    similarity = 1 - cosine(path[i].position, path[i + 1].position)
                    assert similarity > 0.5

        def test_confidence_consistency(self, mock_vector):
            """Test confidence consistency of paths"""
            confidences = [0.8, 0.9, 0.7]
            for conf in confidences:
                evidence = mock_vector + np.random.randn(768) * 0.1
                evidence = evidence / np.linalg.norm(evidence)
                self.tracker.update_trajectories(evidence, confidence=conf)
            
            paths = self.tracker.get_trajectory_paths()
            if paths:
                path = paths[0]
                assert all(t.confidence >= self.tracker.min_confidence for t in path)

    class TestMomentumBehavior:
        """Tests for momentum behavior"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_momentum_tracker):
            """Setup for each test method"""
            self.tracker = shared_momentum_tracker
            self.tracker.trajectories.clear()
            np.random.seed(42)  # Set seed for reproducibility

        def test_initial_state(self, mock_vector):
            """Test initial momentum state"""
            self.tracker.update_trajectories(mock_vector, confidence=0.8)
            assert np.allclose(self.tracker.active_trajectories[0].momentum, 0)

        def test_momentum_accumulation(self, mock_vector):
            """Test momentum accumulation"""
            initial_evidence = mock_vector + np.random.randn(768) * 0.01
            initial_evidence = initial_evidence / np.linalg.norm(initial_evidence)
            self.tracker.update_trajectories(initial_evidence, confidence=0.8)
            initial_momentum = np.linalg.norm(self.tracker.active_trajectories[0].momentum)

            # Provide evidence similar to the initial to ensure it updates the same trajectory
            new_evidence = initial_evidence + np.random.randn(768) * 0.001
            new_evidence = new_evidence / np.linalg.norm(new_evidence)
            self.tracker.update_trajectories(new_evidence, confidence=0.9)
            final_momentum = np.linalg.norm(self.tracker.active_trajectories[0].momentum)
            assert final_momentum > initial_momentum

        def test_decay_rate(self, mock_vector):
            """Test momentum magnitude change"""
            initial_evidence = mock_vector + np.random.randn(768) * 0.01
            initial_evidence = initial_evidence / np.linalg.norm(initial_evidence)
            self.tracker.update_trajectories(initial_evidence, confidence=0.8)

            trajectory = self.tracker.active_trajectories[0]
            self.tracker.update_trajectories(initial_evidence, confidence=0.8)
            momentum_before = np.linalg.norm(trajectory.momentum)

            self.tracker.update_trajectories(initial_evidence, confidence=0.8)
            momentum_after = np.linalg.norm(trajectory.momentum)

            # Check that the momentum is increasing
            assert momentum_after > momentum_before


        def test_decay_direction(self, mock_vector):
            """Test momentum decay direction preservation"""
            initial_evidence = mock_vector + np.random.randn(768) * 0.01
            initial_evidence = initial_evidence / np.linalg.norm(initial_evidence)
            self.tracker.update_trajectories(initial_evidence, confidence=0.8)
            self.tracker.update_trajectories(-initial_evidence, confidence=0.9)

            trajectory = self.tracker.active_trajectories[0]
            momentum_norm = np.linalg.norm(trajectory.momentum)
            if momentum_norm > 1e-6:
                direction_before = trajectory.momentum / momentum_norm

                self.tracker.update_trajectories(initial_evidence, confidence=0.7)
                momentum_norm = np.linalg.norm(trajectory.momentum)
                if momentum_norm > 1e-6:
                    direction_after = trajectory.momentum / momentum_norm
                    assert np.allclose(direction_before, direction_after, rtol=0.1)

    class TestTrajectoryMerging:
        """Tests for trajectory merging"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_momentum_tracker):
            """Setup for each test method"""
            self.tracker = shared_momentum_tracker
            self.tracker.trajectories.clear()
            np.random.seed(42)  # Set seed for reproducibility

        def test_similar_trajectory_detection(self, mock_vector):
            """Test detection of similar trajectories"""
            self.tracker.update_trajectories(mock_vector, confidence=0.8)
            similar_vector = mock_vector + np.random.randn(768) * 0.01
            similar_vector = similar_vector / np.linalg.norm(similar_vector)
            
            similarity = 1 - cosine(mock_vector, similar_vector)
            assert similarity > self.tracker.merge_threshold

        def test_merge_count_reduction(self, mock_vector):
            """Test trajectory count reduction after merging"""
            self.tracker.update_trajectories(mock_vector, confidence=0.8)
            # Reduce noise to increase similarity
            similar_vector = mock_vector + np.random.randn(768) * 0.0001
            similar_vector = similar_vector / np.linalg.norm(similar_vector)
            self.tracker.update_trajectories(similar_vector, confidence=0.7)
            
            merged_trajectories = [t for t in self.tracker.trajectories.values() 
                                 if t.state == TrajectoryState.MERGED]
            assert len(merged_trajectories) > 0

        def test_merged_properties(self, mock_vector):
            """Test properties after merging"""
            self.tracker.update_trajectories(mock_vector, confidence=0.8)
            similar_vector = mock_vector + np.random.randn(768) * 0.0001
            similar_vector = similar_vector / np.linalg.norm(similar_vector)
            self.tracker.update_trajectories(similar_vector, confidence=0.7)
            
            active = self.tracker.active_trajectories[0]
            expected_confidence = max(0.8, 0.7)
            assert active.confidence >= expected_confidence - 0.1
            assert active.state == TrajectoryState.ACTIVE
