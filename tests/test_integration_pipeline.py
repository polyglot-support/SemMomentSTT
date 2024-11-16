"""Tests for the integration pipeline module"""

import pytest
import numpy as np
import torch
from src.integration.pipeline import (
    IntegrationPipeline,
    ProcessingResult,
    NBestHypothesis
)
from src.semantic.momentum_tracker import SemanticTrajectory
from src.decoder.text_decoder import WordScore, DecodingResult

@pytest.fixture
def pipeline():
    """Create an IntegrationPipeline instance for testing"""
    return IntegrationPipeline(
        semantic_dim=768,
        context_window=5,
        n_best=3  # Set N-best size for testing
    )

def test_pipeline_initialization(pipeline):
    """Test that IntegrationPipeline initializes correctly"""
    assert pipeline is not None
    assert pipeline.device in ['cuda', 'cpu']
    assert pipeline.context_window == 5
    assert pipeline.semantic_dim == 768
    assert pipeline.n_best == 3
    assert len(pipeline.context_buffer) == 0
    assert isinstance(pipeline.semantic_projection, torch.nn.Linear)
    assert hasattr(pipeline, 'text_decoder')
    assert pipeline.current_time == 0.0

def test_process_frame(pipeline):
    """Test processing a single frame through the pipeline"""
    # Create a dummy audio frame (16kHz, 1 second)
    audio_frame = np.random.randn(16000).astype(np.float32)
    
    # Process frame with features
    result = pipeline.process_frame(
        audio_frame,
        return_features=True,
        frame_duration=1.0
    )
    
    # Check result structure
    assert isinstance(result, ProcessingResult)
    assert isinstance(result.features, pipeline.acoustic_processor.AcousticFeatures)
    assert 0 <= result.features.confidence <= 1
    
    # Check trajectory
    if result.trajectory is not None:
        assert isinstance(result.trajectory, SemanticTrajectory)
        assert result.trajectory.position.shape == (768,)
        assert 0 <= result.trajectory.confidence <= 1
    
    # Check decoding result
    if result.decoding_result is not None:
        assert isinstance(result.decoding_result, DecodingResult)
        assert isinstance(result.decoding_result.text, str)
        assert 0 <= result.decoding_result.confidence <= 1
        assert len(result.decoding_result.word_scores) == 1
        
        word_score = result.decoding_result.word_scores[0]
        assert isinstance(word_score, WordScore)
        assert word_score.start_time == 0.0  # First frame
        assert word_score.duration == 1.0
    
    # Check N-best results
    assert isinstance(result.n_best, list)
    assert len(result.n_best) <= pipeline.n_best
    for hyp in result.n_best:
        assert isinstance(hyp, NBestHypothesis)
        assert isinstance(hyp.text, str)
        assert 0 <= hyp.confidence <= 1
        assert isinstance(hyp.word_scores, list)
        assert isinstance(hyp.trajectory_path, list)
        assert all(isinstance(t, SemanticTrajectory) for t in hyp.trajectory_path)

def test_n_best_ranking(pipeline):
    """Test ranking of N-best hypotheses"""
    audio_frame = np.random.randn(16000).astype(np.float32)
    result = pipeline.process_frame(audio_frame)
    
    if result.n_best:
        # Check confidence ordering
        confidences = [hyp.confidence for hyp in result.n_best]
        assert confidences == sorted(confidences, reverse=True)
        
        # Best hypothesis should match main result
        best_hyp = result.n_best[0]
        if result.decoding_result is not None:
            assert best_hyp.text == result.decoding_result.text
            assert best_hyp.confidence == result.decoding_result.confidence

def test_time_tracking(pipeline):
    """Test time tracking across frames"""
    frame_duration = 0.5
    n_frames = 5
    
    for i in range(n_frames):
        audio_frame = np.random.randn(8000).astype(np.float32)  # 0.5s at 16kHz
        result = pipeline.process_frame(
            audio_frame,
            frame_duration=frame_duration
        )
        
        # Check time tracking
        expected_time = i * frame_duration
        if result.decoding_result is not None:
            assert result.decoding_result.word_scores[0].start_time == expected_time
            
            # Check N-best timing
            for hyp in result.n_best:
                assert hyp.word_scores[0].start_time == expected_time
    
    assert pipeline.current_time == n_frames * frame_duration

def test_word_history(pipeline):
    """Test word history tracking"""
    frame_duration = 0.5
    n_frames = 10
    
    # Process multiple frames
    for _ in range(n_frames):
        audio_frame = np.random.randn(8000).astype(np.float32)
        pipeline.process_frame(audio_frame, frame_duration=frame_duration)
    
    # Get full history
    history = pipeline.get_word_history()
    if history:
        assert all(isinstance(score, WordScore) for score in history)
        assert all(0 <= score.confidence <= 1 for score in history)
        
        # Check time ordering
        times = [score.start_time for score in history]
        assert times == sorted(times)
    
    # Test time window
    recent = pipeline.get_word_history(time_window=1.0)
    if recent:
        latest_time = recent[-1].start_time
        assert all(score.start_time >= latest_time - 1.0 for score in recent)

def test_process_frame_with_resampling(pipeline):
    """Test processing frames with different sample rates"""
    # Test with 44.1kHz audio
    audio_44k = np.random.randn(44100).astype(np.float32)
    result_44k = pipeline.process_frame(
        audio_44k,
        orig_sr=44100,
        frame_duration=1.0
    )
    assert isinstance(result_44k, ProcessingResult)
    assert isinstance(result_44k.n_best, list)
    
    # Test with 8kHz audio
    audio_8k = np.random.randn(8000).astype(np.float32)
    result_8k = pipeline.process_frame(
        audio_8k,
        orig_sr=8000,
        frame_duration=1.0
    )
    assert isinstance(result_8k, ProcessingResult)
    assert isinstance(result_8k.n_best, list)

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
    assert not np.allclose(context_embedding, 0)

def test_semantic_mapping(pipeline):
    """Test acoustic to semantic space mapping"""
    audio_frame = np.random.randn(16000).astype(np.float32)
    result = pipeline.process_frame(audio_frame, return_features=True)
    
    # Map features to semantic space directly
    semantic_vector = pipeline._map_to_semantic_space(result.features)
    
    assert isinstance(semantic_vector, np.ndarray)
    assert semantic_vector.shape == (768,)
    # Check that vector is normalized
    assert np.abs(np.linalg.norm(semantic_vector) - 1.0) < 1e-6

def test_stream_processing(pipeline):
    """Test processing an audio stream"""
    def dummy_stream(n_frames):
        for _ in range(n_frames):
            yield np.random.randn(8000).astype(np.float32)  # 0.5s frames
    
    # Process a stream of 5 frames
    results = list(pipeline.process_stream(
        dummy_stream(5),
        frame_duration=0.5
    ))
    
    assert len(results) == 5
    for result in results:
        assert isinstance(result, ProcessingResult)
        assert isinstance(result.n_best, list)
        assert len(result.n_best) <= pipeline.n_best

@pytest.mark.parametrize("semantic_dim", [512, 768, 1024])
def test_different_dimensions(semantic_dim):
    """Test pipeline with different semantic dimensions"""
    pipeline = IntegrationPipeline(semantic_dim=semantic_dim)
    
    audio_frame = np.random.randn(16000).astype(np.float32)
    result = pipeline.process_frame(audio_frame)
    
    if result.trajectory is not None:
        assert result.trajectory.position.shape == (semantic_dim,)
        for hyp in result.n_best:
            for traj in hyp.trajectory_path:
                assert traj.position.shape == (semantic_dim,)

def test_n_best_consistency(pipeline):
    """Test consistency of N-best hypotheses"""
    # Process same audio multiple times
    audio_frame = np.random.randn(16000).astype(np.float32)
    
    results = []
    for _ in range(3):
        result = pipeline.process_frame(audio_frame)
        if result.n_best:
            results.append([hyp.text for hyp in result.n_best])
    
    # Should get similar N-best lists
    if len(results) > 1:
        # Compare top hypotheses
        top_hyps = [r[0] for r in results]
        assert len(set(top_hyps)) <= 2  # Allow some variation but not completely random

def test_pipeline_reset(pipeline):
    """Test pipeline reset functionality"""
    # Process some frames
    audio_frame = np.random.randn(16000).astype(np.float32)
    pipeline.process_frame(audio_frame, frame_duration=1.0)
    
    # Reset pipeline
    pipeline.reset()
    
    assert len(pipeline.context_buffer) == 0
    assert pipeline.current_time == 0.0
    assert len(pipeline.get_word_history()) == 0

def test_confidence_propagation(pipeline):
    """Test confidence score propagation"""
    audio_frame = np.random.randn(16000).astype(np.float32)
    result = pipeline.process_frame(audio_frame, frame_duration=1.0)
    
    if result.decoding_result is not None:
        assert 0 <= result.decoding_result.confidence <= 1
        if result.trajectory is not None:
            # Confidence should not exceed trajectory confidence
            assert result.decoding_result.confidence <= result.trajectory.confidence
        
        # Check N-best confidence propagation
        for hyp in result.n_best:
            assert 0 <= hyp.confidence <= 1
            if result.trajectory is not None:
                assert hyp.confidence <= result.trajectory.confidence

def test_trajectory_path_consistency(pipeline):
    """Test consistency of trajectory paths in N-best results"""
    audio_frame = np.random.randn(16000).astype(np.float32)
    result = pipeline.process_frame(audio_frame)
    
    for hyp in result.n_best:
        path = hyp.trajectory_path
        if len(path) > 1:
            # Check temporal ordering
            for i in range(len(path) - 1):
                assert path[i].id < path[i + 1].id
            
            # Last trajectory in path should match confidence
            assert abs(path[-1].confidence - hyp.confidence) < 1e-6
