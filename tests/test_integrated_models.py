"""Integrated testing models combining momentum parameters, audio processing, and visualization"""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from src.main import SemMomentSTT
from src.semantic.momentum_tracker import SemanticTrajectory
from src.integration.pipeline import ProcessingResult, IntegrationPipeline
from src.semantic.types import WordScore

class TestIntegratedModels:
    """Test suite for integrated model validation"""

    @pytest.fixture
    def test_audio(self):
        """Generate test audio data"""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(duration * sample_rate))
        # Generate a 440Hz sine wave with amplitude modulation
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
        audio = audio.astype(np.float32)
        
        test_path = Path("test_audio.wav")
        sf.write(test_path, audio, sample_rate)
        yield test_path
        test_path.unlink()  # Cleanup

    @pytest.fixture
    def optimized_pipeline(self):
        """Create pipeline with optimized momentum parameters"""
        return IntegrationPipeline(
            acoustic_model="facebook/wav2vec2-base",
            language_model="bert-base-uncased",
            semantic_dim=768,
            force_scale=10.0,
            momentum_decay=0.999,
            min_confidence=0.1,  # Lower threshold for testing
            merge_threshold=0.85,
            max_trajectories=5,
            n_best=3
        )

    @pytest.mark.integration
    def test_momentum_trajectory(self, test_audio, optimized_pipeline):
        """Test semantic momentum trajectory with real audio"""
        # Process audio frame by frame
        audio_data, sample_rate = sf.read(test_audio)
        frame_duration = 0.5
        frame_samples = int(frame_duration * sample_rate)
        
        trajectories = []
        confidences = []
        
        for i in range(0, len(audio_data), frame_samples):
            frame = audio_data[i:i + frame_samples]
            if len(frame) == frame_samples:
                # Get acoustic features first
                acoustic_features = optimized_pipeline.acoustic_processor.process_frame(
                    frame,
                    orig_sr=sample_rate
                )
                print(f"\nAcoustic confidence: {acoustic_features.confidence}")
                
                # Map to semantic space
                semantic_vector = optimized_pipeline._map_to_semantic_space(acoustic_features.features)
                print(f"Semantic vector norm: {np.linalg.norm(semantic_vector)}")
                
                # Process frame
                result = optimized_pipeline.process_frame(
                    frame,
                    frame_duration=frame_duration
                )
                
                if result.trajectory is not None:
                    print(f"Trajectory confidence: {result.trajectory.confidence}")
                    trajectories.append(result.trajectory.position)
                    confidences.append(result.trajectory.confidence)
        
        # Validate trajectory properties
        assert len(trajectories) > 0, "No trajectories were generated"
        assert all(isinstance(t, np.ndarray) for t in trajectories)
        assert all(t.shape == (768,) for t in trajectories)
        assert all(0 <= c <= 1 for c in confidences)
        
        # Verify momentum behavior
        if len(trajectories) > 1:
            trajectory_changes = [
                np.linalg.norm(t2 - t1)
                for t1, t2 in zip(trajectories[:-1], trajectories[1:])
            ]
            
            # Momentum should decrease over time
            assert all(c1 >= c2 for c1, c2 in zip(trajectory_changes[:-1], trajectory_changes[1:]))

    @pytest.mark.integration
    def test_n_best_consistency(self, test_audio, optimized_pipeline):
        """Test N-best hypothesis consistency with semantic trajectories"""
        audio_data, sample_rate = sf.read(test_audio)
        
        # Process a few frames to build up context
        frame_duration = 0.5
        frame_samples = int(frame_duration * sample_rate)
        
        for i in range(0, min(len(audio_data), 3 * frame_samples), frame_samples):
            frame = audio_data[i:i + frame_samples]
            if len(frame) == frame_samples:
                result = optimized_pipeline.process_frame(
                    frame,
                    frame_duration=frame_duration
                )
                
                if result.trajectory is not None:
                    print(f"\nFrame {i//frame_samples + 1}:")
                    print(f"Trajectory confidence: {result.trajectory.confidence}")
                    if result.n_best:
                        print(f"N-best hypotheses: {len(result.n_best)}")
                        for idx, hyp in enumerate(result.n_best):
                            print(f"Hypothesis {idx + 1}: {hyp.text} (conf: {hyp.confidence:.3f})")
        
        assert len(result.n_best) > 0, "No N-best hypotheses were generated"
        
        # Verify N-best ordering by confidence
        confidences = [hyp.confidence for hyp in result.n_best]
        assert confidences == sorted(confidences, reverse=True)
        
        # Check trajectory consistency
        for hyp in result.n_best:
            assert len(hyp.trajectory_path) > 0
            assert all(isinstance(t, SemanticTrajectory) for t in hyp.trajectory_path)
            
            # Verify semantic similarity correlates with confidence
            semantic_scores = [
                score.semantic_similarity 
                for score in hyp.word_scores
            ]
            assert all(0 <= score <= 1 for score in semantic_scores)

    @pytest.mark.integration
    def test_lattice_integration(self, test_audio, optimized_pipeline):
        """Test lattice integration with semantic trajectories"""
        audio_data, sample_rate = sf.read(test_audio)
        
        # Process multiple frames to build lattice
        frame_duration = 0.5
        frame_samples = int(frame_duration * sample_rate)
        
        for i in range(0, min(len(audio_data), 3 * frame_samples), frame_samples):
            frame = audio_data[i:i + frame_samples]
            if len(frame) == frame_samples:
                result = optimized_pipeline.process_frame(
                    frame,
                    frame_duration=frame_duration
                )
                
                if result.trajectory is not None:
                    print(f"\nFrame {i//frame_samples + 1}:")
                    print(f"Trajectory confidence: {result.trajectory.confidence}")
                    print(f"Lattice paths: {len(result.lattice_paths)}")
        
        assert len(result.lattice_paths) > 0, "No lattice paths were generated"
        
        # Verify lattice path scores correlate with N-best confidences
        for path, hyp in zip(result.lattice_paths, result.n_best):
            # Path total score should approximately match hypothesis confidence
            assert abs(path.total_score - hyp.confidence) < 1e-6
            
            # Check node sequence matches word sequence
            path_text = " ".join(node.word for node in path.nodes)
            assert path_text == hyp.text

    @pytest.mark.integration
    def test_visualization_generation(self, test_audio, optimized_pipeline):
        """Test visualization generation for semantic trajectories"""
        audio_data, sample_rate = sf.read(test_audio)
        frame_duration = 0.5
        frame_samples = int(frame_duration * sample_rate)
        
        trajectories = []
        timestamps = []
        
        # Collect trajectory data
        for i in range(0, len(audio_data), frame_samples):
            frame = audio_data[i:i + frame_samples]
            if len(frame) == frame_samples:
                result = optimized_pipeline.process_frame(
                    frame,
                    frame_duration=frame_duration
                )
                
                if result.trajectory is not None:
                    print(f"\nFrame {i//frame_samples + 1}:")
                    print(f"Trajectory confidence: {result.trajectory.confidence}")
                    
                    # Use PCA to reduce dimensionality for visualization
                    from sklearn.decomposition import PCA
                    if not trajectories:
                        pca = PCA(n_components=2)
                        trajectories.append(
                            pca.fit_transform(
                                result.trajectory.position.reshape(1, -1)
                            )[0]
                        )
                    else:
                        trajectories.append(
                            pca.transform(
                                result.trajectory.position.reshape(1, -1)
                            )[0]
                        )
                    timestamps.append(i / sample_rate)
        
        assert len(trajectories) > 0, "No trajectories were generated for visualization"
        
        # Generate visualization
        plt.figure(figsize=(10, 6))
        trajectories = np.array(trajectories)
        
        # Plot trajectory
        plt.plot(trajectories[:, 0], trajectories[:, 1], 'b-', label='Semantic Trajectory')
        plt.scatter(trajectories[0, 0], trajectories[0, 1], c='g', label='Start')
        plt.scatter(trajectories[-1, 0], trajectories[-1, 1], c='r', label='End')
        
        plt.title('Semantic Trajectory Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        
        # Save visualization
        plt.savefig('trajectory_visualization.png')
        plt.close()
        
        # Verify visualization was created
        assert Path('trajectory_visualization.png').exists()
        Path('trajectory_visualization.png').unlink()  # Cleanup

    @pytest.mark.integration
    def test_streaming_integration(self, optimized_pipeline):
        """Test streaming integration with momentum tracking"""
        def create_test_stream(duration=2.0, sample_rate=16000):
            chunk_duration = 0.5
            chunk_samples = int(chunk_duration * sample_rate)
            n_chunks = int(duration / chunk_duration)
            
            t = np.linspace(0, chunk_duration, chunk_samples)
            for i in range(n_chunks):
                # Generate a 440Hz sine wave with varying amplitude
                chunk = 0.5 * np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * (t + i * chunk_duration)))
                yield chunk.astype(np.float32)
        
        stream = create_test_stream()
        results = []
        
        for result in optimized_pipeline.process_stream(
            stream,
            frame_duration=0.5
        ):
            results.append(result)
            if result.trajectory is not None:
                print(f"\nStream chunk:")
                print(f"Trajectory confidence: {result.trajectory.confidence}")
                if result.n_best:
                    print(f"N-best hypotheses: {len(result.n_best)}")
        
        assert len(results) > 0, "No results were generated from stream"
        
        # Verify streaming results
        for result in results:
            assert isinstance(result, ProcessingResult)
            if result.trajectory is not None:
                assert isinstance(result.trajectory, SemanticTrajectory)
                assert result.trajectory.position.shape == (768,)
                assert 0 <= result.trajectory.confidence <= 1
            
            # Check N-best hypotheses
            if result.n_best:
                assert all(
                    isinstance(score, WordScore)
                    for hyp in result.n_best
                    for score in hyp.word_scores
                )
                
                # Verify confidence ordering
                confidences = [hyp.confidence for hyp in result.n_best]
                assert confidences == sorted(confidences, reverse=True)
