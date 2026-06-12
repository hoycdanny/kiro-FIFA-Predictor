"""Unit tests for the Dixon-Coles Poisson Model."""

import math

import numpy as np
import pytest

from src.data.data_manager import TeamProfile
from src.engine.dixon_coles import DixonColesModel
from src.utils.constants import SCORE_MATRIX_SIZE


def _make_team(
    name: str = "TeamA",
    confederation: str = "UEFA",
    recent_goals_avg: float = 1.5,
    recent_conceded_avg: float = 1.0,
) -> TeamProfile:
    """Helper to create a minimal TeamProfile for testing."""
    return TeamProfile(
        name=name,
        name_zh="測試隊",
        aliases=[],
        confederation=confederation,
        fifa_ranking=10,
        fifa_points=1700.0,
        elo_rating=1900,
        group="A",
        recent_goals_avg=recent_goals_avg,
        recent_conceded_avg=recent_conceded_avg,
        recent_win_rate=60.0,
        recent_draw_rate=20.0,
        recent_loss_rate=20.0,
        neutral_win_rate=55.0,
        best_wc_result="Quarter-finals",
        vs_top20_win_rate=40.0,
        wc_first_match_win_rate=50.0,
        penalty_shootout_win_rate=50.0,
        first_half_goal_pct=45.0,
        second_half_goal_pct=55.0,
        clean_sheet_rate=30.0,
        failed_to_score_rate=15.0,
    )


class TestDixonColesModel:
    """Tests for DixonColesModel."""

    def setup_method(self):
        self.model = DixonColesModel()

    def test_predict_returns_correct_shape(self):
        """Matrix should be 5×5."""
        team_a = _make_team("France", "UEFA", 1.8, 0.6)
        team_b = _make_team("Brazil", "CONMEBOL", 2.0, 0.8)

        matrix = self.model.predict(team_a, team_b)

        assert matrix.shape == (SCORE_MATRIX_SIZE, SCORE_MATRIX_SIZE)
        assert matrix.shape == (5, 5)

    def test_predict_all_entries_non_negative(self):
        """All entries in the probability matrix should be non-negative."""
        team_a = _make_team("France", "UEFA", 1.8, 0.6)
        team_b = _make_team("Brazil", "CONMEBOL", 2.0, 0.8)

        matrix = self.model.predict(team_a, team_b)

        assert np.all(matrix >= 0)

    def test_predict_matrix_sum_le_one(self):
        """Sum of all probabilities should be ≤ 1.0 (remainder accounts for scores >4)."""
        team_a = _make_team("France", "UEFA", 1.8, 0.6)
        team_b = _make_team("Brazil", "CONMEBOL", 2.0, 0.8)

        matrix = self.model.predict(team_a, team_b)

        total = matrix.sum()
        assert total <= 1.0
        # Should be reasonably close to 1.0 for typical lambda values
        assert total > 0.8

    def test_predict_matrix_sum_reasonable(self):
        """For average teams, the 5×5 matrix should capture most probability mass."""
        team_a = _make_team("Average", "UEFA", 1.35, 1.35)
        team_b = _make_team("Average", "UEFA", 1.35, 1.35)

        matrix = self.model.predict(team_a, team_b)

        # With typical World Cup lambdas (~1.35), 5×5 should cover >90%
        assert matrix.sum() > 0.90

    def test_poisson_probability_basic(self):
        """Verify Poisson probability calculation for known values."""
        model = self.model

        # P(X=0 | λ=1) = e^(-1) ≈ 0.3679
        assert math.isclose(model._poisson_probability(0, 1.0), math.exp(-1), rel_tol=1e-9)

        # P(X=1 | λ=1) = e^(-1) ≈ 0.3679
        assert math.isclose(model._poisson_probability(1, 1.0), math.exp(-1), rel_tol=1e-9)

        # P(X=2 | λ=2) = e^(-2) * 4 / 2 = 2*e^(-2) ≈ 0.2707
        expected = math.exp(-2) * 4 / 2
        assert math.isclose(model._poisson_probability(2, 2.0), expected, rel_tol=1e-9)

    def test_tau_correction_0_0(self):
        """Tau correction for 0-0: 1 - lambda_a * lambda_b * rho."""
        lambda_a, lambda_b, rho = 1.5, 1.2, -0.13
        expected = 1 - lambda_a * lambda_b * rho

        result = self.model._tau_correction(0, 0, lambda_a, lambda_b, rho)

        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_tau_correction_1_0(self):
        """Tau correction for 1-0: 1 + lambda_b * rho."""
        lambda_a, lambda_b, rho = 1.5, 1.2, -0.13
        expected = 1 + lambda_b * rho

        result = self.model._tau_correction(1, 0, lambda_a, lambda_b, rho)

        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_tau_correction_0_1(self):
        """Tau correction for 0-1: 1 + lambda_a * rho."""
        lambda_a, lambda_b, rho = 1.5, 1.2, -0.13
        expected = 1 + lambda_a * rho

        result = self.model._tau_correction(0, 1, lambda_a, lambda_b, rho)

        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_tau_correction_1_1(self):
        """Tau correction for 1-1: 1 - rho."""
        lambda_a, lambda_b, rho = 1.5, 1.2, -0.13
        expected = 1 - rho

        result = self.model._tau_correction(1, 1, lambda_a, lambda_b, rho)

        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_tau_correction_other_scores(self):
        """Tau correction for all other scores should be 1.0."""
        lambda_a, lambda_b, rho = 1.5, 1.2, -0.13

        # Test several other score combinations
        for goals_a, goals_b in [(2, 0), (0, 2), (2, 1), (3, 2), (4, 4)]:
            result = self.model._tau_correction(goals_a, goals_b, lambda_a, lambda_b, rho)
            assert result == 1.0, f"Expected 1.0 for {goals_a}-{goals_b}, got {result}"

    def test_confederation_coefficient_affects_result(self):
        """Teams from stronger confederations should have higher attack strength."""
        team_uefa = _make_team("France", "UEFA", 1.5, 1.0)
        team_ofc = _make_team("NZ", "OFC", 1.5, 1.0)
        opponent = _make_team("Opponent", "UEFA", 1.0, 1.0)

        matrix_uefa = self.model.predict(team_uefa, opponent)
        matrix_ofc = self.model.predict(team_ofc, opponent)

        # UEFA team (coeff 1.05) should have higher expected goals than OFC (coeff 0.85)
        # So UEFA team should have more probability mass in higher goal rows
        # A simple check: sum of prob for >=1 goals by team A
        prob_score_uefa = 1 - matrix_uefa[0, :].sum()
        prob_score_ofc = 1 - matrix_ofc[0, :].sum()

        assert prob_score_uefa > prob_score_ofc
