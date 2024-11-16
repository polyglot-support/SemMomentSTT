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
        merge_threshold=0.85,
        beam_width=3,
        beam_depth=5
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
    assert hasattr(tracker, 'beam_search')
    assert tracker.beam_search.beam_width == 3
    assert tracker.beam_search.max_depth == 5

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
    
    assert len(tracker.active_trajectories) > 0
    trajectory = tracker.active_trajectories[0]
    assert np.allclose(trajectory.position, sample_evidence)
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
    assert not np.allclose(current_position, initial_position)
    
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
    
    initial_count = len(tracker.active_trajectories)
    
    # Merge trajectories
    tracker.merge_similar_trajectories()
    
    # Should have fewer active trajectories
    assert len(tracker.active_trajectories) < initial_count
    assert len([t for t in tracker.trajectories.values() if t.state == TrajectoryState.MERGED]) > 0

def test_trajectory_pruning(tracker, sample_evidence):
    """Test pruning of low confidence trajectories"""
    # Create trajectory with low confidence
    tracker.update_trajectories(sample_evidence, confidence=0.05)
    assert len(tracker.active_trajectories) > 0
    
    # Prune trajectories
    tracker.prune_trajectories()
    assert len(tracker.active_trajectories) == 0
    assert len([t for t in tracker.trajectories.values() if t.state == TrajectoryState.PRUNED]) > 0

def test_beam_search_integration(tracker):
    """Test beam search integration"""
    # Create multiple trajectories
    for _ in range(5):
        evidence = np.random.randn(768)
        evidence = evidence / np.linalg.norm(evidence)
        tracker.update_trajectories(evidence, confidence=0.8)
    
    # Should respect beam width
    assert len(tracker.active_trajectories) <= tracker.beam_search.beam_width
    
    # Get trajectory paths
    paths = tracker.get_trajectory_paths()
    assert isinstance(paths, list)
    assert all(isinstance(path, list) for path in paths)
    assert all(isinstance(traj, SemanticTrajectory) for path in paths for traj in path)

def test_best_trajectory_selection(tracker):
    """Test selection of best trajectory"""
    # Create trajectories with different confidences
    confidences = [0.3, 0.8, 0.5]
    for conf in confidences:
        evidence = np.random.randn(768)
        evidence = evidence / np.linalg.norm(evidence)
        tracker.update_trajectories(evidence, confidence=conf)
    
    best = tracker.get_best_trajectory()
    assert best is not None
    assert best.confidence == max(t.confidence for t in tracker.active_trajectories)

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

def test_path_consistency(tracker):
    """Test consistency of trajectory paths"""
    # Create sequence of related trajectories
    base_vector = np.random.randn(768)
    base_vector = base_vector / np.linalg.norm(base_vector)
    
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

def test_force_field_influence(tracker, sample_evidence):
    """Test influence of force field on trajectories"""
    # Create initial trajectory
    tracker.update_trajectories(sample_evidence, confidence=0.8)
    initial_position = tracker.active_trajectories[0].position.copy()
    
    # Add attraction center very close to trajectory
    tracker.attraction_centers[0] = initial_position + np.random.randn(768) * 0.1
    tracker.attraction_strengths[0] = 2.0  # Strong attraction
    
    # Update several times
    for _ in range(5):
        tracker.update_trajectories(sample_evidence, confidence=0.8)
    
    # Trajectory should be pulled toward attraction center
    final_position = tracker.active_trajectories[0].position
    dist_to_center = np.linalg.norm(final_position - tracker.attraction_centers[0])
    initial_dist = np.linalg.norm(initial_position - tracker.attraction_centers[0])
    assert dist_to_center < initial_dist
