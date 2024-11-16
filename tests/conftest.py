"""PyTest configuration and fixtures"""

import pytest
import numpy as np
import torch

@pytest.fixture
def dummy_audio_frame():
    """Generate a dummy audio frame for testing"""
    # 1 second of audio at 16kHz
    return np.zeros(16000, dtype=np.float32)

@pytest.fixture
def dummy_features():
    """Generate dummy acoustic features for testing"""
    return torch.randn(1, 768)  # Typical Wav2Vec2 feature dimension

@pytest.fixture
def semantic_vector():
    """Generate a dummy semantic vector for testing"""
    return np.random.randn(768)  # Match Wav2Vec2 dimension for consistency

@pytest.fixture
def mock_trajectory():
    """Create a mock trajectory for testing"""
    from src.semantic.momentum_tracker import SemanticTrajectory, TrajectoryState
    
    return SemanticTrajectory(
        id=1,
        position=np.random.randn(768),
        momentum=np.random.randn(768) * 0.1,
        confidence=0.8,
        state=TrajectoryState.ACTIVE,
        history=[np.random.randn(768) for _ in range(5)]
    )
