"""Tests for the text decoder module"""

import pytest
import numpy as np
import torch
from src.decoder.text_decoder import (
    WordScore,
    DecodingResult
)
from src.semantic.types import SemanticTrajectory, TrajectoryState

class TestTextDecoder:
    """Test suite for TextDecoder class"""

    class TestInitialization:
        """Tests for TextDecoder initialization"""

        def test_device_setting(self, shared_text_decoder):
            """Test device configuration"""
            assert shared_text_decoder.device in ['cuda', 'cpu']

        def test_confidence_threshold(self, shared_text_decoder):
            """Test confidence threshold setting"""
            assert shared_text_decoder.min_confidence == 0.3

        def test_context_configuration(self, shared_text_decoder):
            """Test context configuration"""
            assert shared_text_decoder.context_size == 5

        def test_weight_settings(self, shared_text_decoder):
            """Test weight settings"""
            assert shared_text_decoder.lm_weight == 0.3
            assert shared_text_decoder.semantic_weight == 0.7

        def test_model_components(self, shared_text_decoder):
            """Test model component initialization"""
            assert hasattr(shared_text_decoder, 'tokenizer')
            assert hasattr(shared_text_decoder, 'model')

        def test_initial_states(self, shared_text_decoder):
            """Test initial state of collections"""
            assert len(shared_text_decoder.context_tokens) == 0
            assert len(shared_text_decoder.word_history) == 0

    class TestEmbeddings:
        """Tests for token embeddings"""

        def test_embedding_shape(self, shared_text_decoder):
            """Test token embeddings shape"""
            embeddings = shared_text_decoder.token_embeddings
            assert isinstance(embeddings, torch.Tensor)
            assert embeddings.shape[0] == shared_text_decoder.tokenizer.vocab_size
            assert embeddings.shape[1] == 768  # BERT hidden size

        def test_embedding_normalization(self, shared_text_decoder):
            """Test token embeddings normalization"""
            embeddings = shared_text_decoder.token_embeddings
            norms = torch.norm(embeddings, p=2, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    class TestTokenMatching:
        """Tests for token matching functionality"""

        def test_nearest_token_count(self, shared_text_decoder, mock_vector):
            """Test number of nearest tokens returned"""
            tokens = shared_text_decoder._find_nearest_tokens(mock_vector, k=5)
            assert len(tokens) == 5

        def test_token_similarity_range(self, shared_text_decoder, mock_vector):
            """Test similarity scores are in valid range"""
            tokens = shared_text_decoder._find_nearest_tokens(mock_vector, k=5)
            for _, similarity in tokens:
                assert 0 <= similarity <= 1

        def test_similarity_ordering(self, shared_text_decoder, mock_vector):
            """Test tokens are ordered by similarity"""
            tokens = shared_text_decoder._find_nearest_tokens(mock_vector, k=5)
            similarities = [sim for _, sim in tokens]
            assert sorted(similarities, reverse=True) == similarities

    class TestLanguageModel:
        """Tests for language model functionality"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_text_decoder):
            """Setup for each test method"""
            self.decoder = shared_text_decoder
            self.decoder.reset_context()

        def test_basic_scoring(self):
            """Test basic language model scoring"""
            self.decoder.context_tokens = ['the', 'quick', 'brown']
            candidates = [('fox', 0.8), ('dog', 0.7), ('cat', 0.6)]
            
            scored = self.decoder._compute_language_model_scores(candidates)
            assert len(scored) == len(candidates)

        def test_score_ranges(self):
            """Test score ranges are valid"""
            self.decoder.context_tokens = ['the', 'quick', 'brown']
            candidates = [('fox', 0.8), ('dog', 0.7)]
            
            scored = self.decoder._compute_language_model_scores(candidates)
            for _, sem_score, lm_score in scored:
                assert 0 <= sem_score <= 1
                assert 0 <= lm_score <= 1

    class TestTrajectoryDecoding:
        """Tests for trajectory decoding"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_text_decoder):
            """Setup for each test method"""
            self.decoder = shared_text_decoder
            self.decoder.reset_context()

        def test_successful_decoding(self, mock_trajectory):
            """Test successful trajectory decoding"""
            result = self.decoder.decode_trajectory(
                mock_trajectory,
                timestamp=1.0,
                duration=0.5
            )
            
            assert isinstance(result, DecodingResult)
            assert isinstance(result.text, str)
            assert 0 <= result.confidence <= 1
            assert result.trajectory_id == mock_trajectory.id

        def test_word_score_properties(self, mock_trajectory):
            """Test word score properties"""
            result = self.decoder.decode_trajectory(
                mock_trajectory,
                timestamp=1.0,
                duration=0.5
            )
            
            assert len(result.word_scores) == 1
            word_score = result.word_scores[0]
            assert isinstance(word_score, WordScore)
            assert word_score.start_time == 1.0
            assert word_score.duration == 0.5

        def test_low_confidence_handling(self, mock_trajectory):
            """Test handling of low confidence trajectories"""
            mock_trajectory.confidence = 0.2  # Below min_confidence
            result = self.decoder.decode_trajectory(
                mock_trajectory,
                timestamp=1.0,
                duration=0.5
            )
            assert result is None

        def test_confidence_weighting(self, mock_trajectory):
            """Test confidence score weighting"""
            result = self.decoder.decode_trajectory(
                mock_trajectory,
                timestamp=0.0,
                duration=0.5
            )
            
            word_score = result.word_scores[0]
            expected_confidence = (
                self.decoder.semantic_weight * word_score.semantic_similarity +
                self.decoder.lm_weight * word_score.language_model_score
            ) * mock_trajectory.confidence
            
            assert abs(result.confidence - expected_confidence) < 1e-6

    class TestWordHistory:
        """Tests for word history management"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_text_decoder):
            """Setup for each test method"""
            self.decoder = shared_text_decoder
            self.decoder.reset_context()

        def test_history_accumulation(self, mock_trajectory):
            """Test word history accumulation"""
            timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]
            for ts in timestamps:
                self.decoder.decode_trajectory(mock_trajectory, ts, 0.5)
            
            history = self.decoder.get_word_history()
            assert len(history) == len(timestamps)
            assert all(isinstance(score, WordScore) for score in history)

        @pytest.mark.parametrize("time_window", [None, 1.0, 2.0, 5.0])
        def test_time_window_filtering(self, mock_trajectory, time_window):
            """Test word history time window filtering"""
            for i in range(10):
                self.decoder.decode_trajectory(mock_trajectory, i * 0.5, 0.5)
            
            history = self.decoder.get_word_history(time_window)
            
            if time_window is None:
                assert len(history) == 10
            else:
                latest_time = history[-1].start_time if history else 0
                for score in history:
                    assert score.start_time >= (latest_time - time_window)

    class TestContextManagement:
        """Tests for context management"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_text_decoder):
            """Setup for each test method"""
            self.decoder = shared_text_decoder
            self.decoder.reset_context()

        def test_context_size_limit(self, mock_trajectory):
            """Test context size limitation"""
            for i in range(10):  # More than context_size
                self.decoder.decode_trajectory(mock_trajectory, float(i), 0.5)
            
            assert len(self.decoder.context_tokens) <= self.decoder.context_size
            assert len(self.decoder.context_embeddings) <= self.decoder.context_size

        def test_context_string_format(self, mock_trajectory):
            """Test context string formatting"""
            self.decoder.decode_trajectory(mock_trajectory, 0.0, 0.5)
            context = self.decoder.context
            assert isinstance(context, str)
            assert len(context.split()) <= self.decoder.context_size

        def test_reset_functionality(self, mock_trajectory):
            """Test context reset functionality"""
            self.decoder.decode_trajectory(mock_trajectory, 0.0, 0.5)
            assert len(self.decoder.context_tokens) > 0
            
            self.decoder.reset_context()
            assert len(self.decoder.context_tokens) == 0
            assert len(self.decoder.context_embeddings) == 0
            assert len(self.decoder.word_history) == 0
            assert self.decoder.context == ""
