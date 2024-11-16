"""Tests for the beam search module"""

import pytest
import numpy as np
from src.semantic.beam_search import BeamSearch
from src.semantic.types import SemanticTrajectory, TrajectoryState, BeamHypothesis

@pytest.fixture
def beam_search():
    """Create a BeamSearch instance for testing"""
    return BeamSearch(
        beam_width=3,
        max_depth=5,
        score_threshold=0.1,
        diversity_penalty=0.1
    )

@pytest.fixture
def mock_trajectories(mock_trajectory_data):
    """Create mock trajectories for testing"""
    def create_trajectory(id: int, confidence: float) -> SemanticTrajectory:
        return SemanticTrajectory(
            id=id,
            position=mock_trajectory_data['position'],
            momentum=mock_trajectory_data['momentum'],
            confidence=confidence,
            state=TrajectoryState.ACTIVE,
            history=mock_trajectory_data['history']
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

def test_beam_search_initialization(beam_search):
    """Test that BeamSearch initializes correctly"""
    assert beam_search.beam_width == 3
    assert beam_search.max_depth == 5
    assert beam_search.score_threshold == 0.1
    assert beam_search.diversity_penalty == 0.1
    assert len(beam_search.active_beams) == 0
    assert len(beam_search.completed_beams) == 0

def test_hypothesis_scoring(beam_search, mock_trajectories):
    """Test scoring of trajectory hypotheses"""
    # Test initial scoring (no parent)
    score = beam_search.score_hypothesis(mock_trajectories[0][0])
    assert isinstance(score, float)
    assert 0 <= score <= 1
    
    # Test scoring with parent
    parent = BeamHypothesis(
        trajectory=mock_trajectories[0][0],
        score=0.8,
        parent_id=None,
        children_ids=set(),
        depth=0
    )
    score = beam_search.score_hypothesis(mock_trajectories[0][1], parent)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_similarity_computation(beam_search, mock_trajectories):
    """Test trajectory similarity computation"""
    traj1, traj2 = mock_trajectories[0][:2]
    similarity = beam_search._compute_similarity(traj1, traj2)
    
    assert isinstance(similarity, float)
    assert -1 <= similarity <= 1  # Cosine similarity range

def test_beam_update(beam_search, mock_trajectories):
    """Test beam hypothesis updating"""
    # Initial update
    beams = beam_search.update_beams(mock_trajectories[0][:2])
    assert len(beams) <= beam_search.beam_width
    assert all(isinstance(b, BeamHypothesis) for b in beams)
    
    # Second update
    beams = beam_search.update_beams(mock_trajectories[0][2:])
    assert len(beams) <= beam_search.beam_width
    
    # Check beam properties
    for beam in beams:
        assert isinstance(beam.score, float)
        assert 0 <= beam.score <= 1
        assert isinstance(beam.depth, int)
        assert beam.depth >= 0

def test_beam_pruning(beam_search, mock_trajectories):
    """Test pruning of beam hypotheses"""
    # Add some beams
    beam_search.update_beams(mock_trajectories[0])
    
    # Prune with higher threshold
    beam_search.prune_beams(min_score=0.5)
    
    # Check remaining beams
    assert all(b.score >= 0.5 for b in beam_search.active_beams)
    assert all(b.score >= 0.5 for b in beam_search.completed_beams)

def test_path_reconstruction(beam_search, mock_trajectories):
    """Test reconstruction of best path"""
    # Add trajectories in multiple updates
    beam_search.update_beams(mock_trajectories[0][:2])
    beam_search.update_beams(mock_trajectories[0][2:])
    
    # Get best path
    path = beam_search.get_best_path()
    
    assert isinstance(path, list)
    assert all(isinstance(t, SemanticTrajectory) for t in path)
    if path:
        # Path should be ordered by increasing depth
        depths = [beam_search.hypotheses.get(t.id, 0).depth for t in path]
        assert depths == sorted(depths)

def test_beam_width_limit(beam_search, mock_trajectories):
    """Test that beam width limit is respected"""
    # Add more trajectories than beam width
    beams = beam_search.update_beams(mock_trajectories[0])
    assert len(beams) <= beam_search.beam_width

def test_max_depth_limit(beam_search, mock_trajectories):
    """Test that max depth limit is respected"""
    # Perform updates beyond max depth
    for _ in range(beam_search.max_depth + 2):
        beams = beam_search.update_beams(mock_trajectories[0])
    
    assert all(b.depth <= beam_search.max_depth for b in beams)

def test_diversity_penalty(beam_search, mock_vector):
    """Test diversity penalty effect"""
    # Create similar trajectories
    similar_trajectories = []
    for i in range(3):
        # Add small perturbation
        position = mock_vector + np.random.randn(768) * 0.1
        position = position / np.linalg.norm(position)
        
        trajectory = SemanticTrajectory(
            id=i,
            position=position,
            momentum=np.zeros(768),
            confidence=0.8,
            state=TrajectoryState.ACTIVE,
            history=[position]
        )
        similar_trajectories.append(trajectory)
    
    # First trajectory should get full score
    score1 = beam_search.score_hypothesis(similar_trajectories[0])
    
    # Update beams with first trajectory
    beam_search.update_beams([similar_trajectories[0]])
    
    # Similar trajectory should get lower score due to penalty
    score2 = beam_search.score_hypothesis(similar_trajectories[1])
    
    assert score2 < score1

def test_reset(beam_search, mock_trajectories):
    """Test beam search reset"""
    # Add some beams
    beam_search.update_beams(mock_trajectories[0])
    
    # Reset
    beam_search.reset()
    
    assert len(beam_search.active_beams) == 0
    assert len(beam_search.completed_beams) == 0
    assert len(beam_search.hypotheses) == 0

@pytest.mark.parametrize("beam_width", [1, 3, 5])
def test_different_beam_widths(mock_trajectories, beam_width):
    """Test beam search with different beam widths"""
    beam_search = BeamSearch(beam_width=beam_width)
    beams = beam_search.update_beams(mock_trajectories[0])
    assert len(beams) <= beam_width
