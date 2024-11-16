"""Tests for the acoustic processing module"""

import pytest
import numpy as np
from src.acoustic.processor import AcousticProcessor, AcousticFeatures

def test_acoustic_processor_initialization():
    """Test that AcousticProcessor initializes correctly"""
    processor = AcousticProcessor()
    assert processor is not None
    assert processor.device in ['cuda', 'cpu']

def test_process_frame_shape():
    """Test that process_frame returns features with expected shape"""
    processor = AcousticProcessor()
    # Create a dummy audio frame (16kHz, 1 second)
    dummy_frame = np.zeros(16000, dtype=np.float32)
    
    features = processor.process_frame(dummy_frame)
    assert isinstance(features, AcousticFeatures)
    # TODO: Add more specific shape assertions once implementation is complete
