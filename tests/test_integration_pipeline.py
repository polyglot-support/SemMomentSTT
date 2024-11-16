"""Tests for the integration pipeline module"""

import pytest
import numpy as np
import torch
from src.integration.pipeline import (
    ProcessingResult,
    NBestHypothesis
)
from src.semantic.types import SemanticTrajectory
from src.decoder.text_decoder import WordScore, DecodingResult
from src.semantic.lattice import LatticePath

@pytest.mark.integration
def test_pipeline_initialization(shared_pipeline):
    """Test that IntegrationPipeline initializes correctly"""
    pipeline = shared_pipeline
    assert pipeline is not None
    assert pipeline.device in ['cuda', 'cpu']
    assert pipeline.context_window == 10  # Default value
    assert pipeline.semantic_dim == 768
    assert pipeline.n_best == 5  # Default value
    assert len(pipeline.context_buffer) == 0
    assert isinstance(pipeline.semantic_projection, torch.nn.Linear)
    assert hasattr(pipeline, 'text_decoder')
    assert hasattr(pipeline, 'word_lattice')
    assert pipeline.current_time == 0.0

@pytest.mark.integration
def test_process_frame(shared_pipeline, mock_acoustic_features):
    """Test processing a single frame through the pipeline"""
    pipeline = shared_pipeline
    # Reset state
    pipeline.reset()
    
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
    
    # Check lattice paths
    assert isinstance(result.lattice_paths, list)
    for path in result.lattice_paths:
        assert isinstance(path, LatticePath)
        assert path.total_score > 0
        assert len(path.nodes) > 0
        assert len(path.edges) == len(path.nodes) - 1

@pytest.mark.integration
def test_n_best_ranking(shared_pipeline):
    """Test ranking of N-best hypotheses"""
    pipeline = shared_pipeline
    # Reset state
    pipeline.reset()
    
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

@pytest.mark.integration
def test_lattice_generation(shared_pipeline):
    """Test lattice generation and DOT output"""
    pipeline = shared_pipeline
    # Reset state
    pipeline.reset()
    
    # Process multiple frames
    for _ in range(3):
        audio_frame = np.random.randn(16000).astype(np.float32)
        pipeline.process_frame(audio_frame)
    
    # Get lattice visualization
    dot = pipeline.get_lattice_dot()
    assert isinstance(dot, str)
    assert dot.startswith("digraph {")
    assert dot.endswith("}")

@pytest.mark.integration
def test_time_tracking(shared_pipeline):
    """Test time tracking across frames"""
    pipeline = shared_pipeline
    # Reset state
    pipeline.reset()
    
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

@pytest.mark.integration
def test_word_history(shared_pipeline):
    """Test word history tracking"""
    pipeline = shared_pipeline
    # Reset state
    pipeline.reset()
    
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

@pytest.mark.integration
def test_process_frame_with_resampling(shared_pipeline):
    """Test processing frames with different sample rates"""
    pipeline = shared_pipeline
    # Reset state
    pipeline.reset()
    
    # Test with 44.1kHz audio
    audio_44k = np.random.randn(44100).astype(np.float32)
    result_44k = pipeline.process_frame(
        audio_44k,
        orig_sr=44100,
        frame_duration=1.0
    )
    assert isinstance(result_44k, ProcessingResult)
    assert isinstance(result_44k.n_best, list)
    assert isinstance(result_44k.lattice_paths, list)
    
    # Test with 8kHz audio
    audio_8k = np.random.randn(8000).astype(np.float32)
    result_8k = pipeline.process_frame(
        audio_8k,
        orig_sr=8000,
        frame_duration=1.0
    )
    assert isinstance(result_8k, ProcessingResult)
    assert isinstance(result_8k.n_best, list)
    assert isinstance(result_8k.lattice_paths, list)

@pytest.mark.integration
def test_context_management(shared_pipeline):
    """Test context buffer management"""
    pipeline = shared_pipeline
    # Reset state
    pipeline.reset()
    
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

@pytest.mark.integration
def test_semantic_mapping(shared_pipeline, mock_acoustic_features):
    """Test acoustic to semantic space mapping"""
    pipeline = shared_pipeline
    features, _ = mock_acoustic_features
    features = torch.from_numpy(features).float()  # Ensure float32
    
    # Map features to semantic space directly
    semantic_vector = pipeline._map_to_semantic_space(features)
    
    assert isinstance(semantic_vector, np.ndarray)
    assert semantic_vector.shape == (768,)
    # Check that vector is normalized
    assert np.abs(np.linalg.norm(semantic_vector) - 1.0) < 1e-6

@pytest.mark.integration
@pytest.mark.slow
def test_stream_processing(shared_pipeline):
    """Test processing an audio stream"""
    pipeline = shared_pipeline
    # Reset state
    pipeline.reset()
    
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
        assert isinstance(result.lattice_paths, list)
        assert len(result.n_best) <= pipeline.n_best
        assert len(result.lattice_paths) <= pipeline.n_best

@pytest.mark.integration
@pytest.mark.parametrize("semantic_dim", [512, 768, 1024])
def test_different_dimensions(semantic_dim):
    """Test pipeline with different semantic dimensions"""
    from src.integration.pipeline import IntegrationPipeline
    pipeline = IntegrationPipeline(
        acoustic_model="facebook/wav2vec2-base",
        language_model="bert-base-uncased",
        semantic_dim=semantic_dim
    )
    
    audio_frame = np.random.randn(16000).astype(np.float32)
    result = pipeline.process_frame(audio_frame)
    
    if result.trajectory is not None:
        assert result.trajectory.position.shape == (semantic_dim,)
        for hyp in result.n_best:
            for traj in hyp.trajectory_path:
                assert traj.position.shape == (semantic_dim,)

@pytest.mark.integration
def test_pipeline_reset(shared_pipeline):
    """Test pipeline reset functionality"""
    pipeline = shared_pipeline
    # Process some frames
    audio_frame = np.random.randn(16000).astype(np.float32)
    pipeline.process_frame(audio_frame, frame_duration=1.0)
    
    # Reset pipeline
    pipeline.reset()
    
    assert len(pipeline.context_buffer) == 0
    assert pipeline.current_time == 0.0
    assert len(pipeline.get_word_history()) == 0

@pytest.mark.integration
def test_lattice_path_consistency(shared_pipeline):
    """Test consistency between N-best and lattice paths"""
    pipeline = shared_pipeline
    # Reset state
    pipeline.reset()
    
    audio_frame = np.random.randn(16000).astype(np.float32)
    result = pipeline.process_frame(audio_frame)
    
    if result.n_best and result.lattice_paths:
        # N-best and lattice paths should have same number of results
        assert len(result.n_best) == len(result.lattice_paths)
        
        # Best hypothesis should correspond to best lattice path
        best_hyp = result.n_best[0]
        best_path = result.lattice_paths[0]
        
        # Text should match
        path_text = " ".join(node.word for node in best_path.nodes)
        assert best_hyp.text == path_text
        
        # Scores should match
        assert abs(best_hyp.confidence - best_path.total_score) < 1e-6
