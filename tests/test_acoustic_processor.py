"""Tests for the acoustic processing module"""

import pytest
import numpy as np
import torch
from src.acoustic.processor import AcousticProcessor, AcousticFeatures

@pytest.fixture
def processor():
    """Create an AcousticProcessor instance for testing"""
    return AcousticProcessor()

def test_acoustic_processor_initialization(processor):
    """Test that AcousticProcessor initializes correctly"""
    assert processor is not None
    assert processor.device in ['cuda', 'cpu']
    assert processor.sample_rate == 16000
    assert processor.chunk_length == 0.5
    assert processor.chunk_samples == 8000
    assert hasattr(processor, 'model')
    assert hasattr(processor, 'processor')

def test_preprocess_audio(processor):
    """Test audio preprocessing"""
    # Test with float32 audio
    audio = np.random.randn(16000).astype(np.float32)
    processed, duration = processor._preprocess_audio(audio)
    assert isinstance(processed, torch.Tensor)
    assert processed.device.type == processor.device
    assert processed.dim() == 2  # [batch_size, sequence_length]
    assert duration == 1.0  # 1 second
    
    # Test with int16 audio
    audio_int = (audio * 32768).astype(np.int16)
    processed_int, duration = processor._preprocess_audio(audio_int)
    assert isinstance(processed_int, torch.Tensor)
    assert processed_int.dtype == torch.float32
    assert duration == 1.0

def test_resample_audio(processor):
    """Test audio resampling"""
    # Test upsampling
    orig_sr = 8000
    audio = np.random.randn(8000).astype(np.float32)  # 1 second at 8kHz
    resampled, duration = processor._resample_audio(audio, orig_sr)
    assert len(resampled) == 16000  # Should be upsampled to 16kHz
    assert duration == 1.0
    
    # Test downsampling
    orig_sr = 44100
    audio = np.random.randn(44100).astype(np.float32)  # 1 second at 44.1kHz
    resampled, duration = processor._resample_audio(audio, orig_sr)
    assert len(resampled) == 16000  # Should be downsampled to 16kHz
    assert duration == 1.0

def test_process_frame(processor):
    """Test processing a single frame"""
    # Create a 1-second audio frame
    audio_frame = np.random.randn(16000).astype(np.float32)
    
    features = processor.process_frame(audio_frame)
    assert isinstance(features, AcousticFeatures)
    assert isinstance(features.features, torch.Tensor)
    assert features.timestamp == 1.0  # 16000 samples / 16000 Hz = 1 second
    assert features.window_size == 8000
    assert 0 <= features.confidence <= 1

def test_process_frame_with_resampling(processor):
    """Test processing frames with different sample rates"""
    # Test with 44.1kHz audio
    audio_44k = np.random.randn(44100).astype(np.float32)
    features_44k = processor.process_frame(audio_44k, orig_sr=44100)
    assert features_44k.timestamp == 1.0
    
    # Test with 8kHz audio
    audio_8k = np.random.randn(8000).astype(np.float32)
    features_8k = processor.process_frame(audio_8k, orig_sr=8000)
    assert features_8k.timestamp == 1.0

@pytest.mark.parametrize("audio_length", [8000, 16000, 24000])
def test_different_audio_lengths(processor, audio_length):
    """Test processing different audio lengths"""
    audio = np.random.randn(audio_length).astype(np.float32)
    features = processor.process_frame(audio)
    assert isinstance(features, AcousticFeatures)
    assert features.timestamp == audio_length / processor.sample_rate

def test_model_output_shape(processor):
    """Test the shape of model outputs"""
    audio = np.random.randn(16000).astype(np.float32)
    features = processor.process_frame(audio)
    
    # Wav2Vec2 typically reduces the sequence length by a factor of 320
    expected_length = 16000 // 320
    assert features.features.shape[1] == expected_length
    assert features.features.shape[2] == 768  # Base model hidden size

@pytest.mark.parametrize("model_name", [
    "facebook/wav2vec2-base",
    "facebook/wav2vec2-base-960h"
])
def test_different_models(model_name):
    """Test initialization with different Wav2Vec2 models"""
    processor = AcousticProcessor(model_name=model_name)
    assert processor.model_name == model_name
    
    # Test basic functionality
    audio = np.random.randn(16000).astype(np.float32)
    features = processor.process_frame(audio)
    assert isinstance(features, AcousticFeatures)

@pytest.mark.parametrize("sample_rate,duration", [
    (8000, 1.0),
    (16000, 1.0),
    (44100, 1.0),
    (48000, 1.0)
])
def test_various_sample_rates(processor, sample_rate, duration):
    """Test processing audio at various sample rates"""
    samples = int(sample_rate * duration)
    audio = np.random.randn(samples).astype(np.float32)
    
    features = processor.process_frame(audio, orig_sr=sample_rate)
    assert isinstance(features, AcousticFeatures)
    assert np.abs(features.timestamp - duration) < 1e-6  # Allow small numerical errors
