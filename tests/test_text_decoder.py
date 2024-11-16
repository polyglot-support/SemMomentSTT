"""Tests for the text decoder module"""

import pytest
import numpy as np
import torch
from src.decoder.text_decoder import (
    TextDecoder,
    WordScore,
    DecodingResult
)
from src.semantic.momentum_tracker import SemanticTrajectory, TrajectoryState

@pytest.fixture
def decoder():
    """Create a TextDecoder instance for testing"""
    return TextDecoder(
        model_name="bert-base-uncased",
        min_confidence=0.3,
        context_size=3,
        lm_weight=0.3,
        semantic_weight=0.7
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
    assert decoder.lm_weight == 0.3
    assert decoder.semantic_weight == 0.7
    assert hasattr(decoder, 'tokenizer')
    assert hasattr(decoder, 'model')
    assert len(decoder.context_tokens) == 0
    assert len(decoder.word_history) == 0

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
    
    scored = decoder._compute_language_model_scores(candidates)
    
    assert len(scored) == len(candidates)
    for token, sem_score, lm_score in scored:
        assert isinstance(token, str)
        assert 0 <= sem_score <= 1
        assert 0 <= lm_score <= 1

def test_trajectory_decoding(decoder, mock_trajectory):
    """Test decoding a trajectory into text with word scores"""
    result = decoder.decode_trajectory(
        mock_trajectory,
        timestamp=1.0,
        duration=0.5
    )
    
    if result is not None:
        assert isinstance(result, DecodingResult)
        assert isinstance(result.text, str)
        assert 0 <= result.confidence <= 1
        assert result.trajectory_id == mock_trajectory.id
        assert len(result.word_scores) == 1
        
        word_score = result.word_scores[0]
        assert isinstance(word_score, WordScore)
        assert word_score.start_time == 1.0
        assert word_score.duration == 0.5
        assert 0 <= word_score.semantic_similarity <= 1
        assert 0 <= word_score.language_model_score <= 1

def test_low_confidence_trajectory(decoder, mock_trajectory):
    """Test handling of low confidence trajectories"""
    mock_trajectory.confidence = 0.2  # Below min_confidence
    result = decoder.decode_trajectory(
        mock_trajectory,
        timestamp=1.0,
        duration=0.5
    )
    assert result is None

def test_word_history(decoder, mock_trajectory):
    """Test word history management"""
    # Add multiple words
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]
    for ts in timestamps:
        decoder.decode_trajectory(mock_trajectory, ts, 0.5)
    
    # Test full history
    history = decoder.get_word_history()
    assert len(history) == len(timestamps)
    for score in history:
        assert isinstance(score, WordScore)
    
    # Test time window
    recent = decoder.get_word_history(time_window=1.0)
    assert len(recent) == 2  # Should only get last two words
    assert recent[-1].start_time == timestamps[-1]

def test_context_management(decoder, mock_trajectory):
    """Test context management"""
    # Add multiple words
    for i in range(5):  # More than context_size
        decoder.decode_trajectory(mock_trajectory, float(i), 0.5)
    
    assert len(decoder.context_tokens) <= decoder.context_size
    assert len(decoder.context_embeddings) <= decoder.context_size
    
    # Test context string
    context = decoder.context
    assert isinstance(context, str)
    assert len(context.split()) <= decoder.context_size

def test_reset_context(decoder, mock_trajectory):
    """Test context reset"""
    # Add some context
    decoder.decode_trajectory(mock_trajectory, 0.0, 0.5)
    assert len(decoder.context_tokens) > 0
    assert len(decoder.word_history) > 0
    
    # Reset context
    decoder.reset_context()
    assert len(decoder.context_tokens) == 0
    assert len(decoder.context_embeddings) == 0
    assert len(decoder.word_history) == 0
    assert decoder.context == ""

def test_confidence_weighting(decoder, mock_trajectory):
    """Test confidence score weighting"""
    result = decoder.decode_trajectory(mock_trajectory, 0.0, 0.5)
    
    if result is not None:
        word_score = result.word_scores[0]
        
        # Check that final confidence uses weights correctly
        expected_confidence = (
            decoder.semantic_weight * word_score.semantic_similarity +
            decoder.lm_weight * word_score.language_model_score
        ) * mock_trajectory.confidence
        
        assert abs(result.confidence - expected_confidence) < 1e-6

@pytest.mark.parametrize("time_window", [None, 1.0, 2.0, 5.0])
def test_word_history_time_windows(decoder, mock_trajectory, time_window):
    """Test word history with different time windows"""
    # Add words over 5 seconds
    for i in range(10):
        decoder.decode_trajectory(mock_trajectory, i * 0.5, 0.5)
    
    history = decoder.get_word_history(time_window)
    
    if time_window is None:
        assert len(history) == 10
    else:
        # Check that all words are within the time window
        latest_time = history[-1].start_time if history else 0
        for score in history:
            assert score.start_time >= (latest_time - time_window)
