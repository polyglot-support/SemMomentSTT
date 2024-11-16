"""Tests for the integration pipeline module"""

import pytest
import numpy as np
import torch
from src.integration.pipeline import (
    ProcessingResult,
    NBestHypothesis,
    IntegrationPipeline
)
from src.semantic.types import SemanticTrajectory
from src.decoder.text_decoder import WordScore, DecodingResult
from src.semantic.lattice import LatticePath

class TestIntegrationPipeline:
    """Test suite for IntegrationPipeline class"""

    class TestInitialization:
        """Tests for pipeline initialization"""

        @pytest.mark.integration
        def test_basic_attributes(self, shared_pipeline):
            """Test basic pipeline attributes"""
            assert shared_pipeline.device in ['cuda', 'cpu']
            assert shared_pipeline.context_window == 10
            assert shared_pipeline.semantic_dim == 768
            assert shared_pipeline.n_best == 5

        @pytest.mark.integration
        def test_initial_state(self, shared_pipeline):
            """Test initial pipeline state"""
            assert len(shared_pipeline.context_buffer) == 0
            assert shared_pipeline.current_time == 0.0

        @pytest.mark.integration
        def test_component_initialization(self, shared_pipeline):
            """Test component initialization"""
            assert isinstance(shared_pipeline.semantic_projection, torch.nn.Linear)
            assert hasattr(shared_pipeline, 'text_decoder')
            assert hasattr(shared_pipeline, 'word_lattice')

    class TestFrameProcessing:
        """Tests for frame processing functionality"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_pipeline):
            """Setup for each test method"""
            self.pipeline = shared_pipeline
            self.pipeline.reset()

        @pytest.mark.integration
        def test_basic_processing(self, mock_acoustic_features):
            """Test basic frame processing"""
            audio_frame = np.random.randn(16000).astype(np.float32)
            result = self.pipeline.process_frame(
                audio_frame,
                return_features=True,
                frame_duration=1.0
            )
            assert isinstance(result, ProcessingResult)
            assert isinstance(result.features, self.pipeline.acoustic_processor.AcousticFeatures)
            assert 0 <= result.features.confidence <= 1

        @pytest.mark.integration
        def test_trajectory_generation(self, mock_acoustic_features):
            """Test trajectory generation"""
            audio_frame = np.random.randn(16000).astype(np.float32)
            result = self.pipeline.process_frame(audio_frame, frame_duration=1.0)
            
            if result.trajectory is not None:
                assert isinstance(result.trajectory, SemanticTrajectory)
                assert result.trajectory.position.shape == (768,)
                assert 0 <= result.trajectory.confidence <= 1

        @pytest.mark.integration
        def test_decoding_result(self, mock_acoustic_features):
            """Test decoding result generation"""
            audio_frame = np.random.randn(16000).astype(np.float32)
            result = self.pipeline.process_frame(audio_frame, frame_duration=1.0)
            
            if result.decoding_result is not None:
                assert isinstance(result.decoding_result, DecodingResult)
                assert isinstance(result.decoding_result.text, str)
                assert 0 <= result.decoding_result.confidence <= 1
                assert len(result.decoding_result.word_scores) == 1

    class TestNBestProcessing:
        """Tests for N-best processing"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_pipeline):
            """Setup for each test method"""
            self.pipeline = shared_pipeline
            self.pipeline.reset()

        @pytest.mark.integration
        def test_n_best_structure(self):
            """Test N-best hypothesis structure"""
            audio_frame = np.random.randn(16000).astype(np.float32)
            result = self.pipeline.process_frame(audio_frame)
            
            assert isinstance(result.n_best, list)
            assert len(result.n_best) <= self.pipeline.n_best
            
            for hyp in result.n_best:
                assert isinstance(hyp, NBestHypothesis)
                assert isinstance(hyp.text, str)
                assert 0 <= hyp.confidence <= 1
                assert isinstance(hyp.word_scores, list)
                assert isinstance(hyp.trajectory_path, list)

        @pytest.mark.integration
        def test_n_best_ranking(self):
            """Test N-best hypothesis ranking"""
            audio_frame = np.random.randn(16000).astype(np.float32)
            result = self.pipeline.process_frame(audio_frame)
            
            if result.n_best:
                confidences = [hyp.confidence for hyp in result.n_best]
                assert confidences == sorted(confidences, reverse=True)
                
                if result.decoding_result is not None:
                    best_hyp = result.n_best[0]
                    assert best_hyp.text == result.decoding_result.text
                    assert best_hyp.confidence == result.decoding_result.confidence

    class TestLatticeGeneration:
        """Tests for lattice generation"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_pipeline):
            """Setup for each test method"""
            self.pipeline = shared_pipeline
            self.pipeline.reset()

        @pytest.mark.integration
        def test_lattice_paths(self):
            """Test lattice path generation"""
            audio_frame = np.random.randn(16000).astype(np.float32)
            result = self.pipeline.process_frame(audio_frame)
            
            assert isinstance(result.lattice_paths, list)
            for path in result.lattice_paths:
                assert isinstance(path, LatticePath)
                assert path.total_score > 0
                assert len(path.nodes) > 0
                assert len(path.edges) == len(path.nodes) - 1

        @pytest.mark.integration
        def test_dot_visualization(self):
            """Test DOT format generation"""
            for _ in range(3):
                audio_frame = np.random.randn(16000).astype(np.float32)
                self.pipeline.process_frame(audio_frame)
            
            dot = self.pipeline.get_lattice_dot()
            assert isinstance(dot, str)
            assert dot.startswith("digraph {")
            assert dot.endswith("}")

    class TestTimeTracking:
        """Tests for time tracking"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_pipeline):
            """Setup for each test method"""
            self.pipeline = shared_pipeline
            self.pipeline.reset()

        @pytest.mark.integration
        def test_frame_timing(self):
            """Test frame timing tracking"""
            frame_duration = 0.5
            n_frames = 5
            
            for i in range(n_frames):
                audio_frame = np.random.randn(8000).astype(np.float32)
                result = self.pipeline.process_frame(
                    audio_frame,
                    frame_duration=frame_duration
                )
                
                expected_time = i * frame_duration
                if result.decoding_result is not None:
                    assert result.decoding_result.word_scores[0].start_time == expected_time
                    
                    for hyp in result.n_best:
                        assert hyp.word_scores[0].start_time == expected_time
            
            assert self.pipeline.current_time == n_frames * frame_duration

    class TestStreamProcessing:
        """Tests for stream processing"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_pipeline):
            """Setup for each test method"""
            self.pipeline = shared_pipeline
            self.pipeline.reset()

        @pytest.mark.integration
        @pytest.mark.slow
        def test_stream_results(self):
            """Test stream processing results"""
            def dummy_stream(n_frames):
                for _ in range(n_frames):
                    yield np.random.randn(8000).astype(np.float32)
            
            results = list(self.pipeline.process_stream(
                dummy_stream(5),
                frame_duration=0.5
            ))
            
            assert len(results) == 5
            for result in results:
                assert isinstance(result, ProcessingResult)
                assert isinstance(result.n_best, list)
                assert isinstance(result.lattice_paths, list)
                assert len(result.n_best) <= self.pipeline.n_best

    class TestDimensionality:
        """Tests for different semantic dimensions"""

        @pytest.mark.integration
        @pytest.mark.parametrize("semantic_dim", [512, 768, 1024])
        def test_semantic_dimensions(self, semantic_dim):
            """Test different semantic dimensions"""
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

    class TestConsistency:
        """Tests for result consistency"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_pipeline):
            """Setup for each test method"""
            self.pipeline = shared_pipeline
            self.pipeline.reset()

        @pytest.mark.integration
        def test_n_best_lattice_consistency(self):
            """Test consistency between N-best and lattice paths"""
            audio_frame = np.random.randn(16000).astype(np.float32)
            result = self.pipeline.process_frame(audio_frame)
            
            if result.n_best and result.lattice_paths:
                assert len(result.n_best) == len(result.lattice_paths)
                
                best_hyp = result.n_best[0]
                best_path = result.lattice_paths[0]
                
                path_text = " ".join(node.word for node in best_path.nodes)
                assert best_hyp.text == path_text
                assert abs(best_hyp.confidence - best_path.total_score) < 1e-6
