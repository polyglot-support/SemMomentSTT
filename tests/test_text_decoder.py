"""Tests for the text decoder module"""

import pytest
import numpy as np
import torch
from src.decoder.text_decoder import TextDecoder
from src.semantic.momentum_tracker import SemanticTrajectory, TrajectoryState

@pytest.fixture
def decoder():
    """Create a TextDecoder instance for testing"""
    return TextDecoder(
        model_name="bert-base-uncased",
        min_confidence=0.3,
        context_size=3
    )

@pytest.fixture
def mock_trajectory():
    """Create a mock trajectory for testing"""
    return SemanticTrajectory(
        id=1,
        position=np.random.randn(768),  # BERT hidden size
        momentum=np.zeros(768),
        confidence=0.8,
        state=TrajectoryState.ACTIVE,
        history=[np.random.randn(768)]
    )

def test_decoder_initialization(decoder):
    """Test that TextDecoder initializes correctly"""
    assert decoder is not None
    assert decoder.device in ['cuda', 'cpu']
    assert decoder.min_confidence == 0.3
    assert decoder.context_size == 3
    assert hasattr(decoder, 'tokenizer')
    assert hasattr(decoder, 'model')
    assert len(decoder.context_tokens) == 0

def test_token_embeddings_initialization(decoder):
    """Test token embeddings initialization"""
    assert isinstance(decoder.token_embeddings, torch.Tensor)
    assert decoder.token_embeddings.shape[0] == decoder.tokenizer.vocab_size
    assert decoder.token_embeddings.shape[1] == 768  # BERT hidden size
    
    # Check normalization
    norms = torch.norm(decoder.token_embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_find_nearest_tokens(decoder):
    """Test finding nearest tokens"""
    # Create a semantic vector
    vector = np.random.randn(768)
    vector = vector / np.linalg.norm(vector)
    
    # Get nearest tokens
    tokens = decoder._find_nearest_tokens(vector, k=5)
    
    assert len(tokens) == 5
    for token, similarity in tokens:
        assert isinstance(token, str)
        assert 0 <= similarity <= 1
    
    # Check ordering
    similarities = [sim for _, sim in tokens]
    assert sorted(similarities, reverse=True) == similarities

def test_language_model_scoring(decoder):
    """Test language model scoring of candidates"""
    # Add some context
    decoder.context_tokens = ['the', 'quick', 'brown']
    
    candidates = [
        ('fox', 0.8),
        ('dog', 0.7),
        ('cat', 0.6)
    ]
    
    scored = decoder._apply_language_model(candidates)
    
    assert len(scored) == len(candidates)
    for token, score in scored:
        assert isinstance(token, str)
        assert 0 <= score <= 1

def test_trajectory_decoding(decoder, mock_trajectory):
    """Test decoding a trajectory into text"""
    result = decoder.decode_trajectory(mock_trajectory)
    
    if result is not None:
        token, confidence = result
        assert isinstance(token, str)
        assert 0 <= confidence <= 1
        assert len(decoder.context_tokens) == 1
        assert len(decoder.context_embeddings) == 1

def test_low_confidence_trajectory(decoder, mock_trajectory):
    """Test handling of low confidence trajectories"""
    mock_trajectory.confidence = 0.2  # Below min_confidence
    result = decoder.decode_trajectory(mock_trajectory)
    assert result is None

def test_context_management(decoder, mock_trajectory):
    """Test context management"""
    # Add multiple tokens
    for _ in range(5):  # More than context_size
        decoder.decode_trajectory(mock_trajectory)
    
    assert len(decoder.context_tokens) <= decoder.context_size
    assert len(decoder.context_embeddings) <= decoder.context_size
    
    # Test context string
    context = decoder.context
    assert isinstance(context, str)
    assert len(context.split()) <= decoder.context_size

def test_reset_context(decoder, mock_trajectory):
    """Test context reset"""
    # Add some context
    decoder.decode_trajectory(mock_trajectory)
    assert len(decoder.context_tokens) > 0
    
    # Reset context
    decoder.reset_context()
    assert len(decoder.context_tokens) == 0
    assert len(decoder.context_embeddings) == 0
    assert decoder.context == ""

@pytest.mark.parametrize("confidence", [0.4, 0.6, 0.8, 1.0])
def test_different_confidence_levels(decoder, mock_trajectory, confidence):
    """Test decoding with different confidence levels"""
    mock_trajectory.confidence = confidence
    result = decoder.decode_trajectory(mock_trajectory)
    
    assert result is not None
    token, score = result
    assert isinstance(token, str)
    assert score <= confidence  # Final score should not exceed trajectory confidence

def test_consecutive_decoding(decoder, mock_trajectory):
    """Test decoding consecutive trajectories"""
    results = []
    for _ in range(3):
        result = decoder.decode_trajectory(mock_trajectory)
        if result is not None:
            results.append(result)
    
    assert len(results) == 3
    for token, score in results:
        assert isinstance(token, str)
        assert 0 <= score <= 1
