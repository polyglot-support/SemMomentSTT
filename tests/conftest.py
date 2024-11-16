"""Test configuration and shared fixtures"""

import pytest
import numpy as np
from src.acoustic.processor import AcousticProcessor
from src.decoder.text_decoder import TextDecoder
from src.integration.pipeline import IntegrationPipeline
from src.semantic.momentum_tracker import MomentumTracker
from src.semantic.types import SemanticTrajectory, TrajectoryState

@pytest.fixture(scope="session")
def shared_acoustic_processor():
    """Create a shared AcousticProcessor instance for all tests"""
    return AcousticProcessor()

@pytest.fixture(scope="session")
def shared_text_decoder():
    """Create a shared TextDecoder instance for all tests"""
    return TextDecoder()

@pytest.fixture(scope="session")
def shared_pipeline(shared_acoustic_processor, shared_text_decoder):
    """Create a shared Pipeline instance for all tests"""
    return IntegrationPipeline(
        acoustic_model="facebook/wav2vec2-base",
        language_model="bert-base-uncased"
    )

@pytest.fixture(scope="session")
def shared_momentum_tracker():
    """Create a shared MomentumTracker instance for all tests"""
    return MomentumTracker(
        semantic_dim=768,
        max_trajectories=5,
        momentum_decay=0.95,
        min_confidence=0.1,
        merge_threshold=0.85,
        beam_width=3,
        beam_depth=5
    )

@pytest.fixture(autouse=True)
def configure_numpy():
    """Configure numpy printing options for all tests"""
    np.set_printoptions(
        precision=2,
        suppress=True,
        threshold=3,
        edgeitems=2
    )

@pytest.fixture
def mock_vector():
    """Create a normalized random vector for testing"""
    vector = np.random.randn(768)
    return vector / np.linalg.norm(vector)

@pytest.fixture
def mock_trajectory_data():
    """Create mock trajectory data for testing"""
    def create_vector():
        v = np.random.randn(768)
        return v / np.linalg.norm(v)
    
    return {
        'position': create_vector(),
        'momentum': create_vector() * 0.1,
        'confidence': 0.8,
        'history': [create_vector() for _ in range(3)]
    }

@pytest.fixture
def mock_acoustic_features():
    """Create mock acoustic features for testing"""
    features = np.random.randn(1, 10, 768)  # (batch, time, features)
    features = features / np.linalg.norm(features)
    confidence = 0.85
    return features, confidence

@pytest.fixture
def mock_word_scores():
    """Create mock word scores for testing"""
    return [
        ('hello', 0.8, 0.7, 0.9),  # (word, acoustic, lm, semantic)
        ('world', 0.7, 0.8, 0.6),
        ('test', 0.9, 0.8, 0.7)
    ]

@pytest.fixture
def mock_trajectory():
    """Create a mock trajectory for testing"""
    vector = np.random.randn(768)
    vector = vector / np.linalg.norm(vector)
    
    return SemanticTrajectory(
        id=1,
        position=vector,
        momentum=vector * 0.1,
        confidence=0.8,
        state=TrajectoryState.ACTIVE,
        history=[vector.copy()]
    )

@pytest.fixture
def mock_trajectories():
    """Create a list of mock trajectory paths for testing"""
    def create_trajectory(id, confidence):
        vector = np.random.randn(768)
        vector = vector / np.linalg.norm(vector)
        return SemanticTrajectory(
            id=id,
            position=vector,
            momentum=vector * 0.1,
            confidence=confidence,
            state=TrajectoryState.ACTIVE,
            history=[vector.copy()]
        )
    
    # Create two paths, each with 3 trajectories
    paths = [
        # First path with high confidence
        [
            create_trajectory(1, 0.9),
            create_trajectory(2, 0.85),
            create_trajectory(3, 0.8)
        ],
        # Second path with lower confidence
        [
            create_trajectory(4, 0.8),
            create_trajectory(5, 0.75),
            create_trajectory(6, 0.7)
        ]
    ]
    
    return paths

def pytest_configure(config):
    """Custom pytest configuration"""
    # Register custom markers
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )

def pytest_collection_modifyitems(items):
    """Modify test collection to add markers"""
    for item in items:
        # Mark slow tests
        if "test_stream" in item.nodeid or "test_microphone" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

@pytest.fixture(autouse=True)
def _skip_slow(request):
    """Skip slow tests unless explicitly enabled"""
    if request.node.get_closest_marker('slow'):
        if not request.config.getoption("--runslow", default=False):
            pytest.skip('need --runslow option to run')

def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
