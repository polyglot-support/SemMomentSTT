"""Tests for the momentum tracker module"""

import pytest
import numpy as np
from scipy.spatial.distance import cosine
from src.semantic.types import (
    SemanticTrajectory,
    TrajectoryState,
    BeamHypothesis
)

def test_momentum_tracker_initialization(shared_momentum_tracker):
    """Test that MomentumTracker initializes correctly"""
    tracker = shared_momentum_tracker
    assert tracker.semantic_dim == 768
    assert tracker.max_trajectories == 5
    assert tracker.momentum_decay == 0.95
    assert tracker.min_confidence == 0.1
    assert tracker.merge_threshold == 0.85
    assert hasattr(tracker, 'beam_search')
    assert tracker.beam_search.beam_width == 3
    assert tracker.beam_search.max_depth == 5

def test_force_field_computation(shared_momentum_tracker, mock_vector):
    """Test force field computation"""
    tracker = shared_momentum_tracker
    # Reset state
    tracker.trajectories.clear()
    
    force = tracker.compute_force_field(mock_vector)
    
    assert force.shape == mock_vector.shape
    assert isinstance(force, np.ndarray)
    assert not np.allclose(force, 0)  # Force field should not be zero everywhere

def test_trajectory_creation(shared_momentum_tracker, mock_vector):
    """Test creation of new trajectories"""
    tracker = shared_momentum_tracker
    # Reset state
    tracker.trajectories.clear()
    
    tracker.update_trajectories(mock_vector, confidence=0.8)
    
    assert len(tracker.active_trajectories) > 0
    trajectory = tracker.active_trajectories[0]
    assert np.allclose(trajectory.position, mock_vector)
    assert trajectory.confidence == 0.8
    assert trajectory.state == TrajectoryState.ACTIVE

# Breaking down test_trajectory_update into smaller tests
def test_initial_momentum_zero(shared_momentum_tracker, mock_vector):
    """Test that new trajectories start with zero momentum"""
    tracker = shared_momentum_tracker
    tracker.trajectories.clear()
    
    tracker.update_trajectories(mock_vector, confidence=0.8)
    trajectory = tracker.active_trajectories[0]
    
    assert np.allclose(trajectory.momentum, 0)

def test_momentum_builds_up(shared_momentum_tracker, mock_vector):
    """Test that momentum builds up from zero"""
    tracker = shared_momentum_tracker
    tracker.trajectories.clear()
    
    # Create trajectory and update once
    tracker.update_trajectories(mock_vector, confidence=0.8)
    trajectory = tracker.active_trajectories[0]
    initial_momentum = np.linalg.norm(trajectory.momentum)
    
    # Update with opposite direction
    new_evidence = -mock_vector
    tracker.update_trajectories(new_evidence, confidence=0.9)
    final_momentum = np.linalg.norm(trajectory.momentum)
    
    assert final_momentum > initial_momentum

def test_position_changes(shared_momentum_tracker, mock_vector):
    """Test that position changes with updates"""
    tracker = shared_momentum_tracker
    tracker.trajectories.clear()
    
    # Create trajectory
    tracker.update_trajectories(mock_vector, confidence=0.8)
    initial_position = tracker.active_trajectories[0].position.copy()
    
    # Update with new evidence
    new_evidence = mock_vector + np.random.randn(768) * 0.1
    new_evidence = new_evidence / np.linalg.norm(new_evidence)
    tracker.update_trajectories(new_evidence, confidence=0.9)
    
    current_position = tracker.active_trajectories[0].position
    position_delta = np.linalg.norm(current_position - initial_position)
    assert position_delta > 0

# Breaking down test_trajectory_merging into smaller tests
def test_similar_trajectories_detected(shared_momentum_tracker, mock_vector):
    """Test that similar trajectories are detected"""
    tracker = shared_momentum_tracker
    tracker.trajectories.clear()
    
    # Create base trajectory
    tracker.update_trajectories(mock_vector, confidence=0.8)
    
    # Create similar trajectory
    similar_vector = mock_vector + np.random.randn(768) * 0.01  # Very small perturbation
    similar_vector = similar_vector / np.linalg.norm(similar_vector)
    
    # Check similarity
    similarity = 1 - cosine(mock_vector, similar_vector)
    assert similarity > tracker.merge_threshold

def test_trajectory_merging_reduces_count(shared_momentum_tracker, mock_vector):
    """Test that merging reduces trajectory count"""
    tracker = shared_momentum_tracker
    tracker.trajectories.clear()
    
    # Create first trajectory
    tracker.update_trajectories(mock_vector, confidence=0.8)
    
    # Create very similar trajectory
    similar_vector = mock_vector + np.random.randn(768) * 0.01
    similar_vector = similar_vector / np.linalg.norm(similar_vector)
    tracker.update_trajectories(similar_vector, confidence=0.7)
    
    # Check merged state
    merged_trajectories = [t for t in tracker.trajectories.values() 
                         if t.state == TrajectoryState.MERGED]
    assert len(merged_trajectories) > 0

def test_merged_trajectory_properties(shared_momentum_tracker, mock_vector):
    """Test properties of merged trajectories"""
    tracker = shared_momentum_tracker
    tracker.trajectories.clear()
    
    # Create trajectories to merge
    tracker.update_trajectories(mock_vector, confidence=0.8)
    similar_vector = mock_vector + np.random.randn(768) * 0.01
    similar_vector = similar_vector / np.linalg.norm(similar_vector)
    tracker.update_trajectories(similar_vector, confidence=0.7)
    
    # Get surviving trajectory
    active = tracker.active_trajectories[0]
    assert active.confidence >= 0.8  # Should keep highest confidence
    assert active.state == TrajectoryState.ACTIVE

# Breaking down test_momentum_decay into smaller tests
def test_momentum_decay_rate(shared_momentum_tracker, mock_vector):
    """Test that momentum decays at the correct rate"""
    tracker = shared_momentum_tracker
    tracker.trajectories.clear()
    
    # Create and update trajectory to build momentum
    tracker.update_trajectories(mock_vector, confidence=0.8)
    new_evidence = -mock_vector
    tracker.update_trajectories(new_evidence, confidence=0.9)
    
    trajectory = tracker.active_trajectories[0]
    momentum_before = np.linalg.norm(trajectory.momentum)
    
    # Let momentum decay
    tracker.update_trajectories(mock_vector, confidence=0.7)
    momentum_after = np.linalg.norm(trajectory.momentum)
    
    expected_ratio = tracker.momentum_decay
    actual_ratio = momentum_after / momentum_before
    assert np.isclose(actual_ratio, expected_ratio, rtol=0.1)

def test_momentum_decay_direction(shared_momentum_tracker, mock_vector):
    """Test that momentum decay preserves direction"""
    tracker = shared_momentum_tracker
    tracker.trajectories.clear()
    
    # Build momentum
    tracker.update_trajectories(mock_vector, confidence=0.8)
    new_evidence = -mock_vector
    tracker.update_trajectories(new_evidence, confidence=0.9)
    
    trajectory = tracker.active_trajectories[0]
    direction_before = trajectory.momentum / np.linalg.norm(trajectory.momentum)
    
    # Let momentum decay
    tracker.update_trajectories(mock_vector, confidence=0.7)
    direction_after = trajectory.momentum / np.linalg.norm(trajectory.momentum)
    
    assert np.allclose(direction_before, direction_after, rtol=0.1)

# Breaking down test_force_field_influence into smaller tests
def test_attraction_center_creation(shared_momentum_tracker, mock_vector):
    """Test creation of attraction centers"""
    tracker = shared_momentum_tracker
    tracker.trajectories.clear()
    
    # Check attraction centers
    assert tracker.attraction_centers.shape[0] == 10
    assert tracker.attraction_centers.shape[1] == tracker.semantic_dim
    
    # Check normalization
    norms = np.linalg.norm(tracker.attraction_centers, axis=1)
    assert np.allclose(norms, 1.0)

def test_attraction_force_computation(shared_momentum_tracker, mock_vector):
    """Test computation of attraction forces"""
    tracker = shared_momentum_tracker
    tracker.trajectories.clear()
    
    # Create trajectory
    tracker.update_trajectories(mock_vector, confidence=0.8)
    trajectory = tracker.active_trajectories[0]
    
    # Add attraction center
    center_pos = trajectory.position + np.random.randn(768) * 0.1
    center_pos = center_pos / np.linalg.norm(center_pos)
    tracker.attraction_centers[0] = center_pos
    tracker.attraction_strengths[0] = 2.0
    
    # Compute force
    force = tracker.compute_force_field(trajectory.position)
    assert not np.allclose(force, 0)

def test_attraction_influence(shared_momentum_tracker, mock_vector):
    """Test influence of attraction on trajectory"""
    tracker = shared_momentum_tracker
    tracker.trajectories.clear()
    
    # Create trajectory
    tracker.update_trajectories(mock_vector, confidence=0.8)
    trajectory = tracker.active_trajectories[0]
    
    # Add attraction center near trajectory
    center_pos = trajectory.position + np.array([0.1] * 768)
    center_pos = center_pos / np.linalg.norm(center_pos)
    tracker.attraction_centers[0] = center_pos
    tracker.attraction_strengths[0] = 2.0
    
    # Record initial distance
    initial_dist = np.linalg.norm(trajectory.position - center_pos)
    
    # Update several times
    for _ in range(3):
        tracker.update_trajectories(mock_vector, confidence=0.8)
    
    # Check final distance
    final_dist = np.linalg.norm(trajectory.position - center_pos)
    assert final_dist != initial_dist  # Position should change due to attraction

def test_beam_search_integration(shared_momentum_tracker, mock_vector):
    """Test beam search integration"""
    tracker = shared_momentum_tracker
    # Reset state
    tracker.trajectories.clear()
    
    # Create multiple trajectories
    for _ in range(5):
        evidence = mock_vector + np.random.randn(768) * 0.1
        evidence = evidence / np.linalg.norm(evidence)
        tracker.update_trajectories(evidence, confidence=0.8)
    
    # Should respect beam width
    assert len(tracker.active_trajectories) <= tracker.beam_search.beam_width
    
    # Get trajectory paths
    paths = tracker.get_trajectory_paths()
    assert isinstance(paths, list)
    assert all(isinstance(path, list) for path in paths)
    assert all(isinstance(traj, SemanticTrajectory) for path in paths for traj in path)

def test_best_trajectory_selection(shared_momentum_tracker, mock_vector):
    """Test selection of best trajectory"""
    tracker = shared_momentum_tracker
    # Reset state
    tracker.trajectories.clear()
    
    # Create trajectories with different confidences
    confidences = [0.3, 0.8, 0.5]
    for conf in confidences:
        evidence = mock_vector + np.random.randn(768) * 0.1
        evidence = evidence / np.linalg.norm(evidence)
        tracker.update_trajectories(evidence, confidence=conf)
    
    best = tracker.get_best_trajectory()
    assert best is not None
    assert best.confidence == max(t.confidence for t in tracker.active_trajectories)

def test_path_consistency(shared_momentum_tracker, mock_vector):
    """Test consistency of trajectory paths"""
    tracker = shared_momentum_tracker
    # Reset state
    tracker.trajectories.clear()
    
    # Create sequence of related trajectories
    base_vector = mock_vector
    
    for i in range(5):
        # Gradually move the vector
        evidence = base_vector + np.random.randn(768) * 0.1 * i
        evidence = evidence / np.linalg.norm(evidence)
        tracker.update_trajectories(evidence, confidence=0.8)
    
    # Get paths
    paths = tracker.get_trajectory_paths()
    if paths:
        path = paths[0]  # Best path
        
        # Check temporal consistency
        for i in range(len(path) - 1):
            t1, t2 = path[i], path[i + 1]
            # Later trajectories should have higher IDs
            assert t1.id < t2.id
            
            # Positions should be somewhat similar
            similarity = 1 - cosine(t1.position, t2.position)
            assert similarity > 0.5  # Reasonable similarity threshold
