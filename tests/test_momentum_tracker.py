"""Tests for the semantic momentum tracker module"""

import pytest
import numpy as np
from scipy.spatial.distance import cosine
from src.semantic.momentum_tracker import (
    MomentumTracker,
    SemanticTrajectory,
    TrajectoryState
)

@pytest.fixture
def tracker():
    """Create a MomentumTracker instance for testing"""
    return MomentumTracker(
        semantic_dim=768,
        max_trajectories=5,
        momentum_decay=0.95,
        min_confidence=0.1,
        merge_threshold=0.85
    )

@pytest.fixture
def sample_evidence():
    """Generate sample acoustic evidence"""
    # Create normalized random vector
    evidence = np.random.randn(768)
    return evidence / np.linalg.norm(evidence)

def test_momentum_tracker_initialization(tracker):
    """Test that MomentumTracker initializes with correct parameters"""
    assert tracker.semantic_dim == 768
    assert tracker.max_trajectories == 5
    assert tracker.momentum_decay == 0.95
    assert tracker.min_confidence == 0.1
    assert tracker.merge_threshold == 0.85
    assert len(tracker.trajectories) == 0

def test_force_field_computation(tracker):
    """Test force field computation"""
    position = np.zeros(tracker.semantic_dim)
    force = tracker.compute_force_field(position)
    
    assert force.shape == position.shape
    assert isinstance(force, np.ndarray)
    assert not np.allclose(force, 0)  # Force field should not be zero everywhere

def test_trajectory_creation(tracker, sample_evidence):
    """Test creation of new trajectories"""
    tracker.update_trajectories(sample_evidence, confidence=0.8)
    
    assert len(tracker.active_trajectories) == 1
    trajectory = tracker.active_trajectories[0]
    assert np.array_equal(trajectory.position, sample_evidence)
    assert trajectory.confidence == 0.8
    assert trajectory.state == TrajectoryState.ACTIVE

def test_trajectory_update(tracker, sample_evidence):
    """Test trajectory updates with new evidence"""
    # Create initial trajectory
    tracker.update_trajectories(sample_evidence, confidence=0.8)
    initial_position = tracker.active_trajectories[0].position.copy()
    
    # Update with new evidence
    new_evidence = -sample_evidence  # Opposite direction
    tracker.update_trajectories(new_evidence, confidence=0.9)
    
    # Position should have moved
    current_position = tracker.active_trajectories[0].position
    assert not np.array_equal(current_position, initial_position)
    
    # Should have non-zero momentum
    assert not np.allclose(tracker.active_trajectories[0].momentum, 0)

def test_trajectory_merging(tracker):
    """Test that similar trajectories are merged"""
    # Create two similar trajectories
    base_vector = np.random.randn(768)
    base_vector = base_vector / np.linalg.norm(base_vector)
    
    # Slightly perturb the base vector for second trajectory
    perturbed = base_vector + np.random.randn(768) * 0.1
    perturbed = perturbed / np.linalg.norm(perturbed)
    
    # Add both trajectories
    tracker.update_trajectories(base_vector, confidence=0.8)
    tracker.update_trajectories(perturbed, confidence=0.7)
    
    assert len(tracker.active_trajectories) == 2
    
    # Merge trajectories
    tracker.merge_similar_trajectories()
    
    # Should now have only one active trajectory
    assert len(tracker.active_trajectories) == 1
    assert len([t for t in tracker.trajectories.values() if t.state == TrajectoryState.MERGED]) == 1

def test_trajectory_pruning(tracker, sample_evidence):
    """Test pruning of low confidence trajectories"""
    # Create trajectory with low confidence
    tracker.update_trajectories(sample_evidence, confidence=0.05)
    assert len(tracker.active_trajectories) == 1
    
    # Prune trajectories
    tracker.prune_trajectories()
    assert len(tracker.active_trajectories) == 0
    assert len([t for t in tracker.trajectories.values() if t.state == TrajectoryState.PRUNED]) == 1

def test_max_trajectories(tracker):
    """Test that max_trajectories limit is respected"""
    # Create more trajectories than the limit
    for _ in range(10):
        evidence = np.random.randn(768)
        evidence = evidence / np.linalg.norm(evidence)
        tracker.update_trajectories(evidence, confidence=0.8)
        
    assert len(tracker.active_trajectories) <= tracker.max_trajectories

def test_momentum_decay(tracker, sample_evidence):
    """Test that momentum decays over time"""
    # Create trajectory and update it
    tracker.update_trajectories(sample_evidence, confidence=0.8)
    trajectory = tracker.active_trajectories[0]
    
    # Apply several updates
    initial_momentum = np.linalg.norm(trajectory.momentum)
    for _ in range(5):
        tracker.update_trajectories(sample_evidence, confidence=0.8)
    
    final_momentum = np.linalg.norm(trajectory.momentum)
    assert final_momentum < initial_momentum  # Momentum should decay

def test_best_trajectory_selection(tracker):
    """Test selection of best trajectory"""
    # Create multiple trajectories with different confidences
    confidences = [0.3, 0.8, 0.5]
    for conf in confidences:
        evidence = np.random.randn(768)
        evidence = evidence / np.linalg.norm(evidence)
        tracker.update_trajectories(evidence, confidence=conf)
    
    best = tracker.get_best_trajectory()
    assert best is not None
    assert best.confidence == max(confidences)
