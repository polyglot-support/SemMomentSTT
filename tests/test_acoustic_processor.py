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
    processed = processor._preprocess_audio(audio)
    assert isinstance(processed, torch.Tensor)
    assert processed.device.type == processor.device
    assert processed.dim() == 2  # [batch_size, sequence_length]
    
    # Test with int16 audio
    audio_int = (audio * 32768).astype(np.int16)
    processed_int = processor._preprocess_audio(audio_int)
    assert isinstance(processed_int, torch.Tensor)
    assert processed_int.dtype == torch.float32

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
