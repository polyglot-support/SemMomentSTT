"""Tests for the acoustic processing module"""

import pytest
import numpy as np
import torch
from src.acoustic.processor import AcousticFeatures

class TestAcousticProcessor:
    """Test suite for AcousticProcessor class"""

    class TestInitialization:
        """Tests for AcousticProcessor initialization"""

        def test_processor_exists(self, shared_acoustic_processor):
            """Test processor instance creation"""
            assert shared_acoustic_processor is not None

        def test_device_setting(self, shared_acoustic_processor):
            """Test device configuration"""
            assert shared_acoustic_processor.device in ['cuda', 'cpu']

        def test_sample_rate_config(self, shared_acoustic_processor):
            """Test sample rate configuration"""
            assert shared_acoustic_processor.sample_rate == 16000

        def test_chunk_settings(self, shared_acoustic_processor):
            """Test chunk configuration"""
            assert shared_acoustic_processor.chunk_length == 0.5
            assert shared_acoustic_processor.chunk_samples == 8000

        def test_required_attributes(self, shared_acoustic_processor):
            """Test presence of required model attributes"""
            assert hasattr(shared_acoustic_processor, 'model')
            assert hasattr(shared_acoustic_processor, 'processor')

    class TestPreprocessing:
        """Tests for audio preprocessing functionality"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_acoustic_processor):
            """Setup for each test method"""
            self.processor = shared_acoustic_processor
            self.processor.reset()

        def test_float32_preprocessing(self):
            """Test preprocessing of float32 audio data"""
            audio = np.random.randn(16000).astype(np.float32)
            processed, duration = self.processor._preprocess_audio(audio)
            
            assert isinstance(processed, torch.Tensor)
            assert processed.device.type == self.processor.device
            assert processed.dim() == 2
            assert duration == 1.0

        def test_int16_preprocessing(self):
            """Test preprocessing of int16 audio data"""
            audio = (np.random.randn(16000) * 32768).astype(np.int16)
            processed, duration = self.processor._preprocess_audio(audio)
            
            assert isinstance(processed, torch.Tensor)
            assert processed.dtype == torch.float32
            assert duration == 1.0

    class TestResampling:
        """Tests for audio resampling functionality"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_acoustic_processor):
            """Setup for each test method"""
            self.processor = shared_acoustic_processor
            self.processor.reset()

        @pytest.mark.parametrize("orig_sr,samples", [
            (8000, 8000),    # Upsampling
            (44100, 44100),  # Downsampling
            (48000, 48000)   # Downsampling
        ])
        def test_resampling_rates(self, orig_sr, samples):
            """Test resampling from various sample rates"""
            audio = np.random.randn(samples).astype(np.float32)
            resampled = self.processor._resample_audio(audio, orig_sr)
            assert len(resampled) == 16000

        def test_resampling_preserves_duration(self):
            """Test that resampling preserves audio duration"""
            orig_sr = 44100
            duration = 1.0  # 1 second
            samples = int(orig_sr * duration)
            audio = np.random.randn(samples).astype(np.float32)
            resampled = self.processor._resample_audio(audio, orig_sr)
            assert len(resampled) == 16000  # Target sample rate * duration

    class TestFrameProcessing:
        """Tests for frame processing functionality"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_acoustic_processor):
            """Setup for each test method"""
            self.processor = shared_acoustic_processor
            self.processor.reset()

        def test_basic_frame_processing(self):
            """Test basic frame processing functionality"""
            audio_frame = np.random.randn(16000).astype(np.float32)
            features = self.processor.process_frame(audio_frame)
            
            assert isinstance(features, AcousticFeatures)
            assert isinstance(features.features, torch.Tensor)
            assert features.timestamp == 0.0

        def test_frame_confidence(self):
            """Test confidence score calculation"""
            audio_frame = np.random.randn(16000).astype(np.float32)
            features = self.processor.process_frame(audio_frame)
            assert 0 <= features.confidence <= 1

        @pytest.mark.parametrize("audio_length", [8000, 16000, 24000])
        def test_variable_frame_lengths(self, audio_length):
            """Test processing frames of different lengths"""
            audio = np.random.randn(audio_length).astype(np.float32)
            features = self.processor.process_frame(audio)
            
            assert isinstance(features, AcousticFeatures)
            assert features.timestamp == 0.0

    class TestModelOutput:
        """Tests for model output characteristics"""

        @pytest.fixture(autouse=True)
        def setup_method(self, shared_acoustic_processor):
            """Setup for each test method"""
            self.processor = shared_acoustic_processor
            self.processor.reset()

        def test_output_tensor_shape(self):
            """Test the shape of model output tensors"""
            audio = np.random.randn(16000).astype(np.float32)
            features = self.processor.process_frame(audio)
            
            expected_length = 16000 // 320  # Wav2Vec2 reduction factor
            assert features.features.shape[1] == expected_length
            assert features.features.shape[2] == 768  # Base model hidden size

        @pytest.mark.parametrize("sample_rate,duration", [
            (8000, 1.0),
            (16000, 1.0),
            (44100, 1.0),
            (48000, 1.0)
        ])
        def test_output_with_various_sample_rates(self, sample_rate, duration):
            """Test model output with various input sample rates"""
            samples = int(sample_rate * duration)
            audio = np.random.randn(samples).astype(np.float32)
            
            features = self.processor.process_frame(audio, orig_sr=sample_rate)
            assert isinstance(features, AcousticFeatures)
            assert features.timestamp == 0.0
