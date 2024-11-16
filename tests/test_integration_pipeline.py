"""Tests for the integration pipeline module"""

import pytest
import numpy as np
import torch
from src.integration.pipeline import IntegrationPipeline
from src.semantic.momentum_tracker import SemanticTrajectory
from src.acoustic.processor import AcousticFeatures

@pytest.fixture
def pipeline():
    """Create an IntegrationPipeline instance for testing"""
    return IntegrationPipeline(
        semantic_dim=768,
        context_window=5
    )

def test_pipeline_initialization(pipeline):
    """Test that IntegrationPipeline initializes correctly"""
    assert pipeline is not None
    assert pipeline.device in ['cuda', 'cpu']
    assert pipeline.context_window == 5
    assert pipeline.semantic_dim == 768
    assert len(pipeline.context_buffer) == 0
    assert isinstance(pipeline.semantic_projection, torch.nn.Linear)

def test_process_frame(pipeline):
    """Test processing a single frame through the pipeline"""
    # Create a dummy audio frame (16kHz, 1 second)
    audio_frame = np.random.randn(16000).astype(np.float32)
    
    # Process frame and get both trajectory and features
    trajectory, features = pipeline.process_frame(audio_frame, return_features=True)
    
    # Check acoustic features
    assert isinstance(features, AcousticFeatures)
    assert isinstance(features.features, torch.Tensor)
    assert 0 <= features.confidence <= 1
    
    # Initially there should be a trajectory
    assert isinstance(trajectory, SemanticTrajectory)
    assert trajectory.position.shape == (768,)
    assert 0 <= trajectory.confidence <= 1

def test_context_management(pipeline):
    """Test context buffer management"""
    # Process multiple frames
    for _ in range(10):  # More than context window
        audio_frame = np.random.randn(16000).astype(np.float32)
        pipeline.process_frame(audio_frame)
    
    # Check context buffer size
    assert len(pipeline.context_buffer) == pipeline.context_window
    
    # Check context embedding
    context_embedding = pipeline.get_context_embedding()
    assert isinstance(context_embedding, np.ndarray)
    assert context_embedding.shape == (768,)
    assert not np.allclose(context_embedding, 0)  # Should not be zero vector

def test_semantic_mapping(pipeline):
    """Test acoustic to semantic space mapping"""
    audio_frame = np.random.randn(16000).astype(np.float32)
    trajectory, features = pipeline.process_frame(audio_frame, return_features=True)
    
    # Map features to semantic space directly
    semantic_vector = pipeline._map_to_semantic_space(features)
    
    assert isinstance(semantic_vector, np.ndarray)
    assert semantic_vector.shape == (768,)
    # Check that vector is normalized
    assert np.abs(np.linalg.norm(semantic_vector) - 1.0) < 1e-6

def test_stream_processing(pipeline):
    """Test processing an audio stream"""
    def dummy_stream(n_frames):
        for _ in range(n_frames):
            yield np.random.randn(16000).astype(np.float32)
    
    # Process a stream of 5 frames
    trajectories = list(pipeline.process_stream(dummy_stream(5)))
    
    assert len(trajectories) == 5
    for trajectory in trajectories:
        assert isinstance(trajectory, SemanticTrajectory)
        assert trajectory.position.shape == (768,)

@pytest.mark.parametrize("semantic_dim", [512, 768, 1024])
def test_different_dimensions(semantic_dim):
    """Test pipeline with different semantic dimensions"""
    pipeline = IntegrationPipeline(semantic_dim=semantic_dim)
    
    audio_frame = np.random.randn(16000).astype(np.float32)
    trajectory = pipeline.process_frame(audio_frame)
    
    assert trajectory.position.shape == (semantic_dim,)

def test_confidence_propagation(pipeline):
    """Test that confidence scores are properly propagated"""
    # Process same frame multiple times to build confidence
    audio_frame = np.random.randn(16000).astype(np.float32)
    
    confidences = []
    for _ in range(5):
        trajectory = pipeline.process_frame(audio_frame)
        confidences.append(trajectory.confidence)
    
    # Confidence should stabilize
    assert len(set(confidences)) > 1  # Should not be constant
    assert 0 <= min(confidences) <= max(confidences) <= 1

def test_trajectory_consistency(pipeline):
    """Test that trajectories maintain consistency over time"""
    # Process similar frames
    base_frame = np.random.randn(16000).astype(np.float32)
    
    # Get initial trajectory
    first_trajectory = pipeline.process_frame(base_frame)
    initial_position = first_trajectory.position.copy()
    
    # Process slightly perturbed versions
    for _ in range(5):
        perturbed = base_frame + np.random.randn(16000) * 0.1
        trajectory = pipeline.process_frame(perturbed)
        
        # Position should not change dramatically
        assert np.linalg.norm(trajectory.position - initial_position) < 2.0
