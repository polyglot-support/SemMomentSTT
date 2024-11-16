"""Tests for the beam search module"""

import pytest
import numpy as np
from src.semantic.beam_search import BeamSearch
from src.semantic.types import SemanticTrajectory, TrajectoryState, BeamHypothesis

class TestBeamSearch:
    """Test suite for BeamSearch class"""

    @pytest.fixture
    def beam_search(self):
        """Create a BeamSearch instance for testing"""
        return BeamSearch(
            beam_width=3,
            max_depth=5,
            score_threshold=0.1,
            diversity_penalty=0.1
        )

    @pytest.fixture
    def mock_trajectories(self, mock_trajectory_data):
        """Create mock trajectories for testing"""
        def create_trajectory(id: int, confidence: float) -> SemanticTrajectory:
            return SemanticTrajectory(
                id=id,
                position=mock_trajectory_data['position'],
                momentum=mock_trajectory_data['momentum'],
                confidence=confidence,
                state=TrajectoryState.ACTIVE,
                history=mock_trajectory_data['history']
            )
        
        # Create two paths of trajectories
        path1 = [
            create_trajectory(1, 0.8),
            create_trajectory(2, 0.7),
            create_trajectory(3, 0.9)
        ]
        path2 = [
            create_trajectory(4, 0.6),
            create_trajectory(5, 0.8),
            create_trajectory(6, 0.7)
        ]
        
        return [path1, path2]

    class TestInitialization:
        """Tests for BeamSearch initialization"""

        def test_beam_width_setting(self, beam_search):
            """Test beam width configuration"""
            assert beam_search.beam_width == 3

        def test_max_depth_setting(self, beam_search):
            """Test max depth configuration"""
            assert beam_search.max_depth == 5

        def test_score_threshold_setting(self, beam_search):
            """Test score threshold configuration"""
            assert beam_search.score_threshold == 0.1

        def test_diversity_penalty_setting(self, beam_search):
            """Test diversity penalty configuration"""
            assert beam_search.diversity_penalty == 0.1

        def test_initial_beam_states(self, beam_search):
            """Test initial beam states"""
            assert len(beam_search.active_beams) == 0
            assert len(beam_search.completed_beams) == 0

    class TestHypothesisScoring:
        """Tests for hypothesis scoring functionality"""

        def test_initial_hypothesis_scoring(self, beam_search, mock_trajectories):
            """Test scoring of initial hypothesis without parent"""
            score = beam_search.score_hypothesis(mock_trajectories[0][0])
            assert isinstance(score, float)
            assert 0 <= score <= 1

        def test_child_hypothesis_scoring(self, beam_search, mock_trajectories):
            """Test scoring of hypothesis with parent"""
            parent = BeamHypothesis(
                trajectory=mock_trajectories[0][0],
                score=0.8,
                parent_id=None,
                children_ids=set(),
                depth=0
            )
            score = beam_search.score_hypothesis(mock_trajectories[0][1], parent)
            assert isinstance(score, float)
            assert 0 <= score <= 1

        def test_similarity_computation(self, beam_search, mock_trajectories):
            """Test trajectory similarity computation"""
            traj1, traj2 = mock_trajectories[0][:2]
            similarity = beam_search._compute_similarity(traj1, traj2)
            assert isinstance(similarity, float)
            assert -1 <= similarity <= 1

    class TestBeamManagement:
        """Tests for beam management functionality"""

        def test_initial_beam_update(self, beam_search, mock_trajectories):
            """Test initial beam update"""
            beams = beam_search.update_beams(mock_trajectories[0][:2])
            assert len(beams) <= beam_search.beam_width
            assert all(isinstance(b, BeamHypothesis) for b in beams)

        def test_subsequent_beam_update(self, beam_search, mock_trajectories):
            """Test subsequent beam update"""
            beam_search.update_beams(mock_trajectories[0][:2])
            beams = beam_search.update_beams(mock_trajectories[0][2:])
            assert len(beams) <= beam_search.beam_width

        def test_beam_hypothesis_properties(self, beam_search, mock_trajectories):
            """Test properties of beam hypotheses"""
            beams = beam_search.update_beams(mock_trajectories[0])
            for beam in beams:
                assert isinstance(beam.score, float)
                assert 0 <= beam.score <= 1
                assert isinstance(beam.depth, int)
                assert beam.depth >= 0

        def test_beam_pruning_above_threshold(self, beam_search, mock_trajectories):
            """Test pruning beams above threshold"""
            beam_search.update_beams(mock_trajectories[0])
            beam_search.prune_beams(min_score=0.5)
            assert all(b.score >= 0.5 for b in beam_search.active_beams)

        def test_beam_pruning_completed_beams(self, beam_search, mock_trajectories):
            """Test pruning of completed beams"""
            beam_search.update_beams(mock_trajectories[0])
            beam_search.prune_beams(min_score=0.5)
            assert all(b.score >= 0.5 for b in beam_search.completed_beams)

    class TestPathReconstruction:
        """Tests for path reconstruction functionality"""

        def test_best_path_structure(self, beam_search, mock_trajectories):
            """Test structure of reconstructed best path"""
            beam_search.update_beams(mock_trajectories[0])
            path = beam_search.get_best_path()
            assert isinstance(path, list)
            assert all(isinstance(t, SemanticTrajectory) for t in path)

        def test_path_depth_ordering(self, beam_search, mock_trajectories):
            """Test depth ordering of reconstructed path"""
            beam_search.update_beams(mock_trajectories[0][:2])
            beam_search.update_beams(mock_trajectories[0][2:])
            path = beam_search.get_best_path()
            if path:
                depths = [beam_search.hypotheses.get(t.id, 0).depth for t in path]
                assert depths == sorted(depths)

    class TestConstraints:
        """Tests for beam search constraints"""

        @pytest.mark.parametrize("beam_width", [1, 3, 5])
        def test_beam_width_constraint(self, mock_trajectories, beam_width):
            """Test beam width constraint with different widths"""
            beam_search = BeamSearch(beam_width=beam_width)
            beams = beam_search.update_beams(mock_trajectories[0])
            assert len(beams) <= beam_width

        def test_max_depth_constraint(self, beam_search, mock_trajectories):
            """Test max depth constraint"""
            for _ in range(beam_search.max_depth + 2):
                beams = beam_search.update_beams(mock_trajectories[0])
            assert all(b.depth <= beam_search.max_depth for b in beams)

    class TestDiversityPenalty:
        """Tests for diversity penalty functionality"""

        def test_diversity_penalty_effect(self, beam_search, mock_vector):
            """Test effect of diversity penalty on similar trajectories"""
            # Create similar trajectories
            similar_trajectories = []
            for i in range(3):
                position = mock_vector + np.random.randn(768) * 0.1
                position = position / np.linalg.norm(position)
                trajectory = SemanticTrajectory(
                    id=i,
                    position=position,
                    momentum=np.zeros(768),
                    confidence=0.8,
                    state=TrajectoryState.ACTIVE,
                    history=[position]
                )
                similar_trajectories.append(trajectory)

            score1 = beam_search.score_hypothesis(similar_trajectories[0])
            beam_search.update_beams([similar_trajectories[0]])
            score2 = beam_search.score_hypothesis(similar_trajectories[1])
            assert score2 < score1

    class TestStateManagement:
        """Tests for beam search state management"""

        def test_reset_active_beams(self, beam_search, mock_trajectories):
            """Test reset of active beams"""
            beam_search.update_beams(mock_trajectories[0])
            beam_search.reset()
            assert len(beam_search.active_beams) == 0

        def test_reset_completed_beams(self, beam_search, mock_trajectories):
            """Test reset of completed beams"""
            beam_search.update_beams(mock_trajectories[0])
            beam_search.reset()
            assert len(beam_search.completed_beams) == 0

        def test_reset_hypotheses(self, beam_search, mock_trajectories):
            """Test reset of hypotheses"""
            beam_search.update_beams(mock_trajectories[0])
            beam_search.reset()
            assert len(beam_search.hypotheses) == 0

class TestBeamSearchIntegration:
    """Integration tests for BeamSearch"""

    def test_momentum_tracker_integration(self, shared_momentum_tracker, mock_vector):
        """Test integration with momentum tracker"""
        tracker = shared_momentum_tracker
        tracker.trajectories.clear()

        # Create and track trajectories
        for _ in range(5):
            evidence = mock_vector + np.random.randn(768) * 0.1
            evidence = evidence / np.linalg.norm(evidence)
            tracker.update_trajectories(evidence, confidence=0.8)

        # Verify beam width constraint
        assert len(tracker.active_trajectories) <= tracker.beam_search.beam_width

        # Verify trajectory paths
        paths = tracker.get_trajectory_paths()
        assert isinstance(paths, list)
        assert all(isinstance(path, list) for path in paths)
        assert all(isinstance(traj, SemanticTrajectory) for path in paths for traj in path)

        # Verify path properties
        if paths:
            path = paths[0]
            assert len(path) > 0
            assert all(t.state == TrajectoryState.ACTIVE for t in path)
            assert all(t.confidence >= tracker.min_confidence for t in path)
