"""Unit tests for the Monte Carlo tournament simulator."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.engine.monte_carlo import (
    MonteCarloSimulator,
    TournamentResult,
    ChampionPrediction,
)


def _make_team_profile(name: str, elo: int = 1800, penalty_rate: float = 50.0):
    """Create a minimal mock TeamProfile for testing."""
    profile = MagicMock()
    profile.name = name
    profile.elo_rating = elo
    profile.penalty_shootout_win_rate = penalty_rate
    profile.recent_goals_avg = 1.5
    profile.recent_conceded_avg = 1.0
    profile.confederation = "UEFA"
    profile.current_win_streak = 0
    profile.current_loss_streak = 0
    profile.last_match_date = None
    profile.eliminated_by_2022 = None
    return profile


def _make_mock_ensemble():
    """Create a mock ensemble model that returns fixed probabilities."""
    ensemble = MagicMock()
    # Default: roughly even match with slight home advantage
    ensemble.predict.return_value = (0.40, 0.20, 0.40)
    return ensemble


def _create_32_teams():
    """Generate 32 team names for testing."""
    return [f"Team_{i:02d}" for i in range(32)]


def _create_teams_dict(team_names):
    """Create a teams dictionary with mock profiles."""
    return {name: _make_team_profile(name) for name in team_names}


class TestMonteCarloSimulatorInit:
    """Tests for MonteCarloSimulator initialization."""

    def test_default_n_simulations(self):
        ensemble = _make_mock_ensemble()
        teams = _create_teams_dict(["A", "B"])
        sim = MonteCarloSimulator(ensemble, teams)
        assert sim.n_simulations == 10000

    def test_custom_n_simulations(self):
        ensemble = _make_mock_ensemble()
        teams = _create_teams_dict(["A", "B"])
        sim = MonteCarloSimulator(ensemble, teams, n_simulations=500)
        assert sim.n_simulations == 500


class TestSimulateTournament:
    """Tests for the full tournament simulation."""

    def test_returns_valid_champion_prediction(self):
        """Simulation with small n (100) returns valid ChampionPrediction."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()

        sim = MonteCarloSimulator(ensemble, teams, n_simulations=100)
        result = sim.simulate_tournament(team_names)

        assert isinstance(result, ChampionPrediction)
        assert result.predicted_champion in team_names

    def test_champion_probability_between_0_and_1(self):
        """Champion probability is between 0 and 1."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()

        sim = MonteCarloSimulator(ensemble, teams, n_simulations=100)
        result = sim.simulate_tournament(team_names)

        assert 0.0 <= result.champion_probability <= 1.0

    def test_all_round_probabilities_between_0_and_1(self):
        """All round probabilities are between 0 and 1."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()

        sim = MonteCarloSimulator(ensemble, teams, n_simulations=100)
        result = sim.simulate_tournament(team_names)

        for team, rounds in result.round_probabilities.items():
            for round_name, prob in rounds.items():
                assert 0.0 <= prob <= 1.0, (
                    f"Probability for {team} in {round_name} is {prob}, "
                    f"expected between 0 and 1"
                )

    def test_confidence_index_in_range(self):
        """confidence_index is in [0, 100]."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()

        sim = MonteCarloSimulator(ensemble, teams, n_simulations=100)
        result = sim.simulate_tournament(team_names)

        assert 0 <= result.confidence_index <= 100
        assert isinstance(result.confidence_index, int)

    def test_top_5_has_correct_structure(self):
        """Top 5 list contains up to 5 tuples of (team, probability)."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()

        sim = MonteCarloSimulator(ensemble, teams, n_simulations=100)
        result = sim.simulate_tournament(team_names)

        assert len(result.top_5) <= 5
        for team, prob in result.top_5:
            assert team in team_names
            assert 0.0 <= prob <= 1.0

    def test_top_5_sorted_descending(self):
        """Top 5 is sorted by probability in descending order."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()

        sim = MonteCarloSimulator(ensemble, teams, n_simulations=100)
        result = sim.simulate_tournament(team_names)

        probs = [prob for _, prob in result.top_5]
        assert probs == sorted(probs, reverse=True)

    def test_raises_error_for_non_32_teams(self):
        """Raises ValueError if not exactly 32 teams."""
        ensemble = _make_mock_ensemble()
        teams = _create_teams_dict(["A", "B", "C"])
        sim = MonteCarloSimulator(ensemble, teams, n_simulations=10)

        with pytest.raises(ValueError, match="Expected 32 qualified teams"):
            sim.simulate_tournament(["A", "B", "C"])

    def test_all_32_teams_have_round_probabilities(self):
        """All 32 teams appear in round_probabilities."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()

        sim = MonteCarloSimulator(ensemble, teams, n_simulations=100)
        result = sim.simulate_tournament(team_names)

        assert set(result.round_probabilities.keys()) == set(team_names)

    def test_round_probabilities_are_non_increasing(self):
        """For each team, probability of advancing decreases with later rounds."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()

        sim = MonteCarloSimulator(ensemble, teams, n_simulations=200)
        result = sim.simulate_tournament(team_names)

        round_order = MonteCarloSimulator.ROUND_NAMES
        for team, rounds in result.round_probabilities.items():
            for i in range(len(round_order) - 1):
                # Later rounds should have <= probability than earlier rounds
                assert rounds[round_order[i]] >= rounds[round_order[i + 1]] - 0.01, (
                    f"{team}: {round_order[i]}={rounds[round_order[i]]} < "
                    f"{round_order[i+1]}={rounds[round_order[i+1]]}"
                )


class TestSimulateSingleMatch:
    """Tests for the _resolve_knockout_match method."""

    def test_returns_team_a_on_low_random(self):
        """Low random value → team A wins."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()
        # predict returns (0.40, 0.20, 0.40)
        sim = MonteCarloSimulator(ensemble, teams, n_simulations=10)
        sim._prob_cache[("Team_00", "Team_01")] = (0.40, 0.20, 0.40)

        # random_value = 0.1 < 0.40 → team A wins
        winner = sim._resolve_knockout_match("Team_00", "Team_01", 0.1)
        assert winner == "Team_00"

    def test_returns_team_b_on_high_random(self):
        """High random value → team B wins."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()
        sim = MonteCarloSimulator(ensemble, teams, n_simulations=10)
        sim._prob_cache[("Team_00", "Team_01")] = (0.40, 0.20, 0.40)

        # random_value = 0.9 > 0.60 → team B wins
        winner = sim._resolve_knockout_match("Team_00", "Team_01", 0.9)
        assert winner == "Team_01"

    def test_draw_resolves_to_valid_team(self):
        """Draw range → resolves via penalty shootout to a valid team."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()
        sim = MonteCarloSimulator(ensemble, teams, n_simulations=10)
        sim._prob_cache[("Team_00", "Team_01")] = (0.40, 0.20, 0.40)

        # random_value = 0.5 → in draw range [0.40, 0.60)
        winner = sim._resolve_knockout_match("Team_00", "Team_01", 0.5)
        assert winner in ("Team_00", "Team_01")


class TestCalculateConfidence:
    """Tests for the confidence calculation."""

    def test_high_confidence_with_dominant_team(self):
        """When one team dominates, confidence is high."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()
        sim = MonteCarloSimulator(ensemble, teams, n_simulations=100)

        # Simulate: one team wins 30% of the time, next two 20% each
        champion_counts = {name: 0 for name in team_names}
        champion_counts["Team_00"] = 30
        champion_counts["Team_01"] = 20
        champion_counts["Team_02"] = 20
        # Distribute rest
        remaining = 100 - 70
        for i in range(3, 32):
            champion_counts[f"Team_{i:02d}"] = remaining // 29

        confidence = sim._calculate_confidence([], champion_counts)
        assert 0 <= confidence <= 100
        # Top 1 is 30%, top 3 is 70% → should be high confidence
        assert confidence >= 80

    def test_low_confidence_with_even_distribution(self):
        """When results are spread evenly, confidence is low."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()
        sim = MonteCarloSimulator(ensemble, teams, n_simulations=100)

        # Even distribution: each team wins ~3 times
        champion_counts = {name: 3 for name in team_names}
        # Adjust to total = 100
        champion_counts["Team_00"] = 3 + (100 - 3 * 32)

        confidence = sim._calculate_confidence([], champion_counts)
        assert 0 <= confidence <= 100
        # Very spread out → low confidence
        assert confidence <= 50

    def test_confidence_always_in_valid_range(self):
        """Confidence is always between 0 and 100 regardless of input."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()
        sim = MonteCarloSimulator(ensemble, teams, n_simulations=100)

        # Edge case: all zero
        champion_counts = {name: 0 for name in team_names}
        confidence = sim._calculate_confidence([], champion_counts)
        assert 0 <= confidence <= 100

        # Edge case: one team wins all
        champion_counts["Team_00"] = 100
        confidence = sim._calculate_confidence([], champion_counts)
        assert 0 <= confidence <= 100


class TestWithDominantTeam:
    """Test with a biased ensemble that favors certain teams."""

    def test_strong_team_wins_more_often(self):
        """A team with higher win probability should win more often."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = MagicMock()

        # Make Team_00 stronger against everyone
        def biased_predict(team_a, team_b):
            if team_a.name == "Team_00":
                return (0.70, 0.15, 0.15)
            elif team_b.name == "Team_00":
                return (0.15, 0.15, 0.70)
            return (0.40, 0.20, 0.40)

        ensemble.predict.side_effect = biased_predict

        sim = MonteCarloSimulator(ensemble, teams, n_simulations=500)
        result = sim.simulate_tournament(team_names)

        # Team_00 should be the predicted champion (or at least in top 5)
        top_5_names = [name for name, _ in result.top_5]
        assert "Team_00" in top_5_names


class TestWithCustomBracket:
    """Tests for bracket format handling."""

    def test_matches_bracket_format(self):
        """Bracket with 'matches' key works correctly."""
        team_names = _create_32_teams()
        teams = _create_teams_dict(team_names)
        ensemble = _make_mock_ensemble()

        bracket = {
            "matches": [[i, 31 - i] for i in range(16)]
        }

        sim = MonteCarloSimulator(ensemble, teams, n_simulations=50)
        result = sim.simulate_tournament(team_names, bracket=bracket)

        assert isinstance(result, ChampionPrediction)
        assert result.predicted_champion in team_names
