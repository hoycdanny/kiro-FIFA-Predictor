"""
Unit tests for the Ensemble Model.

Tests verify:
1. Default weights validate correctly
2. Invalid weights fail validation
3. redistribute_without produces valid weights summing to 1.0
4. predict returns probabilities summing to 1.0
5. predict_with_fallback handles model failure gracefully
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.data_manager import TeamProfile
from src.engine.ensemble import (
    AllModelsFailedError,
    EnsembleModel,
    EnsembleWeights,
)


def _make_team(
    name: str = "TeamA",
    elo_rating: int = 1800,
    confederation: str = "UEFA",
    goals_avg: float = 1.5,
    conceded_avg: float = 1.0,
    win_streak: int = 0,
    loss_streak: int = 0,
    last_match_date: str | None = None,
    eliminated_by_2022: str | None = None,
) -> TeamProfile:
    """Helper to create a minimal TeamProfile for testing."""
    return TeamProfile(
        name=name,
        name_zh="",
        aliases=[],
        confederation=confederation,
        fifa_ranking=1,
        fifa_points=1500.0,
        elo_rating=elo_rating,
        group="A",
        recent_goals_avg=goals_avg,
        recent_conceded_avg=conceded_avg,
        recent_win_rate=50.0,
        recent_draw_rate=25.0,
        recent_loss_rate=25.0,
        neutral_win_rate=45.0,
        best_wc_result="Quarter-finals",
        vs_top20_win_rate=30.0,
        wc_first_match_win_rate=50.0,
        penalty_shootout_win_rate=50.0,
        first_half_goal_pct=45.0,
        second_half_goal_pct=55.0,
        clean_sheet_rate=30.0,
        failed_to_score_rate=15.0,
        current_win_streak=win_streak,
        current_loss_streak=loss_streak,
        last_match_date=last_match_date,
        eliminated_by_2022=eliminated_by_2022,
    )


class TestEnsembleWeightsValidation:
    """Test EnsembleWeights.validate() method."""

    def test_default_weights_validate(self):
        """Default weights (0.40, 0.25, 0.15, 0.20) should pass validation."""
        weights = EnsembleWeights()
        assert weights.validate() is True

    def test_custom_valid_weights(self):
        """Custom valid weights within [0.10, 0.60] summing to 1.0."""
        weights = EnsembleWeights(poisson=0.30, elo=0.30, h2h=0.20, dynamic=0.20)
        assert weights.validate() is True

    def test_boundary_min_weights(self):
        """Weights at minimum boundary (0.10) for some models."""
        weights = EnsembleWeights(poisson=0.60, elo=0.10, h2h=0.10, dynamic=0.20)
        assert weights.validate() is True

    def test_boundary_max_weights(self):
        """Weights at maximum boundary (0.60) for one model."""
        weights = EnsembleWeights(poisson=0.60, elo=0.15, h2h=0.10, dynamic=0.15)
        assert weights.validate() is True

    def test_weight_below_minimum_fails(self):
        """Weight below WEIGHT_MIN (0.10) should fail."""
        weights = EnsembleWeights(poisson=0.45, elo=0.30, h2h=0.05, dynamic=0.20)
        assert weights.validate() is False

    def test_weight_above_maximum_fails(self):
        """Weight above WEIGHT_MAX (0.60) should fail."""
        weights = EnsembleWeights(poisson=0.65, elo=0.15, h2h=0.10, dynamic=0.10)
        assert weights.validate() is False

    def test_weights_not_summing_to_one_fails(self):
        """Weights not summing to 1.0 should fail."""
        weights = EnsembleWeights(poisson=0.40, elo=0.25, h2h=0.15, dynamic=0.25)
        assert weights.validate() is False

    def test_all_weights_at_zero_fails(self):
        """All weights at zero fails both range and sum checks."""
        weights = EnsembleWeights(poisson=0.0, elo=0.0, h2h=0.0, dynamic=0.0)
        assert weights.validate() is False


class TestEnsembleWeightsRedistribute:
    """Test EnsembleWeights.redistribute_without() method."""

    def test_exclude_poisson_sums_to_one(self):
        """Excluding poisson redistributes to remaining, sum = 1.0."""
        weights = EnsembleWeights()
        new_weights = weights.redistribute_without("poisson")
        total = new_weights.elo + new_weights.h2h + new_weights.dynamic
        assert math.isclose(total, 1.0, abs_tol=1e-9)
        assert new_weights.poisson == 0.0

    def test_exclude_elo_sums_to_one(self):
        """Excluding elo redistributes to remaining, sum = 1.0."""
        weights = EnsembleWeights()
        new_weights = weights.redistribute_without("elo")
        total = new_weights.poisson + new_weights.h2h + new_weights.dynamic
        assert math.isclose(total, 1.0, abs_tol=1e-9)
        assert new_weights.elo == 0.0

    def test_exclude_h2h_sums_to_one(self):
        """Excluding h2h redistributes to remaining, sum = 1.0."""
        weights = EnsembleWeights()
        new_weights = weights.redistribute_without("h2h")
        total = new_weights.poisson + new_weights.elo + new_weights.dynamic
        assert math.isclose(total, 1.0, abs_tol=1e-9)
        assert new_weights.h2h == 0.0

    def test_exclude_dynamic_sums_to_one(self):
        """Excluding dynamic redistributes to remaining, sum = 1.0."""
        weights = EnsembleWeights()
        new_weights = weights.redistribute_without("dynamic")
        total = new_weights.poisson + new_weights.elo + new_weights.h2h
        assert math.isclose(total, 1.0, abs_tol=1e-9)
        assert new_weights.dynamic == 0.0

    def test_redistribute_preserves_proportions(self):
        """Remaining weights maintain their original proportions."""
        weights = EnsembleWeights(poisson=0.40, elo=0.25, h2h=0.15, dynamic=0.20)
        new_weights = weights.redistribute_without("poisson")

        # Original remaining: elo=0.25, h2h=0.15, dynamic=0.20 (sum=0.60)
        # Expected: elo=0.25/0.60, h2h=0.15/0.60, dynamic=0.20/0.60
        assert math.isclose(new_weights.elo, 0.25 / 0.60, abs_tol=1e-9)
        assert math.isclose(new_weights.h2h, 0.15 / 0.60, abs_tol=1e-9)
        assert math.isclose(new_weights.dynamic, 0.20 / 0.60, abs_tol=1e-9)

    def test_redistribute_invalid_model_raises(self):
        """Excluding an unknown model name raises ValueError."""
        weights = EnsembleWeights()
        with pytest.raises(ValueError):
            weights.redistribute_without("nonexistent")


class TestEnsembleModelPredict:
    """Test EnsembleModel.predict() method."""

    def test_predict_returns_probabilities_summing_to_one(self):
        """Predict should return (win_a, draw, win_b) summing to 1.0."""
        model = EnsembleModel()
        team_a = _make_team("Brazil", elo_rating=2000, goals_avg=2.0, conceded_avg=0.8)
        team_b = _make_team("Japan", elo_rating=1700, goals_avg=1.3, conceded_avg=1.2)

        win_a, draw, win_b = model.predict(team_a, team_b)
        assert math.isclose(win_a + draw + win_b, 1.0, abs_tol=1e-6)

    def test_predict_all_probabilities_non_negative(self):
        """All probabilities should be non-negative."""
        model = EnsembleModel()
        team_a = _make_team("Germany", elo_rating=1900, goals_avg=1.8, conceded_avg=0.9)
        team_b = _make_team("Mexico", elo_rating=1750, goals_avg=1.4, conceded_avg=1.1)

        win_a, draw, win_b = model.predict(team_a, team_b)
        assert win_a >= 0.0
        assert draw >= 0.0
        assert win_b >= 0.0

    def test_predict_stronger_team_has_higher_win_prob(self):
        """A significantly stronger team should have higher win probability."""
        model = EnsembleModel()
        team_a = _make_team("France", elo_rating=2100, goals_avg=2.5, conceded_avg=0.5)
        team_b = _make_team("Jamaica", elo_rating=1400, goals_avg=0.8, conceded_avg=2.0)

        win_a, draw, win_b = model.predict(team_a, team_b)
        assert win_a > win_b

    def test_predict_equal_teams_balanced(self):
        """Equal teams should produce roughly balanced probabilities."""
        model = EnsembleModel()
        team_a = _make_team("TeamA", elo_rating=1800)
        team_b = _make_team("TeamB", elo_rating=1800)

        win_a, draw, win_b = model.predict(team_a, team_b)
        # Should be roughly equal (not exact due to H2H and dynamic factors)
        assert abs(win_a - win_b) < 0.15


class TestEnsembleModelPredictWithFallback:
    """Test EnsembleModel.predict_with_fallback() method."""

    def test_fallback_with_no_failures_sums_to_one(self):
        """When all models succeed, result sums to 1.0."""
        model = EnsembleModel()
        team_a = _make_team("Brazil", elo_rating=2000, goals_avg=2.0, conceded_avg=0.8)
        team_b = _make_team("Japan", elo_rating=1700, goals_avg=1.3, conceded_avg=1.2)

        win_a, draw, win_b = model.predict_with_fallback(team_a, team_b)
        assert math.isclose(win_a + draw + win_b, 1.0, abs_tol=1e-6)

    def test_fallback_handles_single_model_failure(self):
        """When one model fails, result still sums to 1.0."""
        model = EnsembleModel()
        # Make Dixon-Coles raise an exception
        model.dixon_coles.predict = MagicMock(side_effect=RuntimeError("Model failed"))

        team_a = _make_team("Brazil", elo_rating=2000, goals_avg=2.0, conceded_avg=0.8)
        team_b = _make_team("Japan", elo_rating=1700, goals_avg=1.3, conceded_avg=1.2)

        win_a, draw, win_b = model.predict_with_fallback(team_a, team_b)
        assert math.isclose(win_a + draw + win_b, 1.0, abs_tol=1e-6)
        assert win_a >= 0.0
        assert draw >= 0.0
        assert win_b >= 0.0

    def test_fallback_handles_multiple_model_failures(self):
        """When multiple models fail, result still valid if at least one succeeds."""
        model = EnsembleModel()
        # Fail Dixon-Coles and H2H
        model.dixon_coles.predict = MagicMock(side_effect=RuntimeError("Failed"))
        model.h2h_model.predict = MagicMock(side_effect=RuntimeError("Failed"))

        team_a = _make_team("Brazil", elo_rating=2000, goals_avg=2.0, conceded_avg=0.8)
        team_b = _make_team("Japan", elo_rating=1700, goals_avg=1.3, conceded_avg=1.2)

        win_a, draw, win_b = model.predict_with_fallback(team_a, team_b)
        assert math.isclose(win_a + draw + win_b, 1.0, abs_tol=1e-6)

    def test_fallback_all_models_fail_raises_error(self):
        """When all models fail, AllModelsFailedError is raised."""
        model = EnsembleModel()
        model.dixon_coles.predict = MagicMock(side_effect=RuntimeError("Failed"))
        model.elo_model.predict = MagicMock(side_effect=RuntimeError("Failed"))
        model.h2h_model.predict = MagicMock(side_effect=RuntimeError("Failed"))
        model.dynamic_factor.calculate_adjustment = MagicMock(
            side_effect=RuntimeError("Failed")
        )

        team_a = _make_team("Brazil", elo_rating=2000, goals_avg=2.0, conceded_avg=0.8)
        team_b = _make_team("Japan", elo_rating=1700, goals_avg=1.3, conceded_avg=1.2)

        with pytest.raises(AllModelsFailedError):
            model.predict_with_fallback(team_a, team_b)

    def test_fallback_probabilities_non_negative(self):
        """All probabilities from fallback are non-negative."""
        model = EnsembleModel()
        # Fail one model
        model.elo_model.predict = MagicMock(side_effect=ValueError("Elo failed"))

        team_a = _make_team("Germany", elo_rating=1900, goals_avg=1.8, conceded_avg=0.9)
        team_b = _make_team("Canada", elo_rating=1650, goals_avg=1.2, conceded_avg=1.3)

        win_a, draw, win_b = model.predict_with_fallback(team_a, team_b)
        assert win_a >= 0.0
        assert draw >= 0.0
        assert win_b >= 0.0


class TestEnsembleModelUpdateWeights:
    """Test EnsembleModel.update_weights() method."""

    def test_update_with_valid_weights(self):
        """Updating with valid weights succeeds."""
        model = EnsembleModel()
        new_weights = EnsembleWeights(poisson=0.35, elo=0.30, h2h=0.15, dynamic=0.20)
        model.update_weights(new_weights)
        assert model.weights == new_weights

    def test_update_with_invalid_weights_raises(self):
        """Updating with invalid weights raises ValueError."""
        model = EnsembleModel()
        invalid_weights = EnsembleWeights(poisson=0.70, elo=0.10, h2h=0.10, dynamic=0.10)
        with pytest.raises(ValueError):
            model.update_weights(invalid_weights)
