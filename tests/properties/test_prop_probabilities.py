"""
Property-based tests for prediction engine probability invariants.

Tests Properties 1, 2, and 3 from the design document:
- Property 1: Win/Draw/Lose probability sum invariant
- Property 2: Over/Under probability sum invariant
- Property 3: Confidence index range invariant

**Validates: Requirements 1.2, 1.3, 1.4**
"""

from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis.strategies import sampled_from

from src.data.data_manager import DataManager
from src.engine.ensemble import EnsembleModel
from src.engine.prediction_engine import PredictionEngine


# Use real data directory
DATA_DIR = Path(__file__).parents[2] / "data"

# Coach styles to test
COACH_STYLES = ["analyst", "contrarian", "tactician"]

# Load actual team names from the data file to avoid mismatches
_data_manager = DataManager(data_dir=DATA_DIR)
_teams = _data_manager.load_teams()
AVAILABLE_TEAMS = sorted(set(t.name for t in _teams))


@pytest.fixture(scope="module")
def engine():
    """Initialize PredictionEngine once for all tests in this module."""
    data_manager = DataManager(data_dir=DATA_DIR)
    ensemble = EnsembleModel()
    return PredictionEngine(data_manager=data_manager, ensemble=ensemble)


class TestWinDrawLoseProbabilitySum:
    """Property 1: Win/Draw/Lose probability sum invariant.

    For any valid pair of teams and for any coach style (analyst, contrarian,
    tactician), the predicted win + draw + lose probabilities SHALL equal
    100.0% (within ±0.1%).

    **Validates: Requirements 1.2**
    """

    @given(
        team_a=sampled_from(AVAILABLE_TEAMS),
        team_b=sampled_from(AVAILABLE_TEAMS),
        style=sampled_from(COACH_STYLES),
    )
    @settings(max_examples=50, deadline=None)
    def test_wdl_sum_equals_100(self, engine, team_a, team_b, style):
        """W + D + L must sum to 1.0 (100%) within ±0.1% tolerance."""
        assume(team_a != team_b)

        prediction = engine.predict_match(team_a, team_b, coach_style=style)

        total = prediction.win_prob + prediction.draw_prob + prediction.lose_prob

        assert abs(total - 1.0) <= 0.001, (
            f"W+D+L = {total:.6f} for {team_a} vs {team_b} (style={style}). "
            f"Expected 1.0 ± 0.001. "
            f"W={prediction.win_prob}, D={prediction.draw_prob}, L={prediction.lose_prob}"
        )


class TestOverUnderProbabilitySum:
    """Property 2: Over/Under probability sum invariant.

    For any valid pair of teams, the predicted over-2.5-goals probability +
    under-2.5-goals probability SHALL equal 100.0% (within ±0.1%).

    **Validates: Requirements 1.3**
    """

    @given(
        team_a=sampled_from(AVAILABLE_TEAMS),
        team_b=sampled_from(AVAILABLE_TEAMS),
    )
    @settings(max_examples=50, deadline=None)
    def test_over_under_sum_equals_100(self, engine, team_a, team_b):
        """Over 2.5 + Under 2.5 must sum to 1.0 (100%) within ±0.1% tolerance."""
        assume(team_a != team_b)

        prediction = engine.predict_match(team_a, team_b)

        total = prediction.over_2_5 + prediction.under_2_5

        assert abs(total - 1.0) <= 0.001, (
            f"Over+Under = {total:.6f} for {team_a} vs {team_b}. "
            f"Expected 1.0 ± 0.001. "
            f"Over={prediction.over_2_5}, Under={prediction.under_2_5}"
        )


class TestConfidenceIndexRange:
    """Property 3: Confidence index range invariant.

    For any prediction output (single match or tournament simulation),
    the confidence_index SHALL be an integer in the range [0, 100].

    **Validates: Requirements 1.4**
    """

    @given(
        team_a=sampled_from(AVAILABLE_TEAMS),
        team_b=sampled_from(AVAILABLE_TEAMS),
        style=sampled_from(COACH_STYLES),
    )
    @settings(max_examples=50, deadline=None)
    def test_confidence_index_is_valid_integer_in_range(self, engine, team_a, team_b, style):
        """Confidence index must be an integer in [0, 100]."""
        assume(team_a != team_b)

        prediction = engine.predict_match(team_a, team_b, coach_style=style)

        assert isinstance(prediction.confidence_index, int), (
            f"confidence_index is {type(prediction.confidence_index)}, expected int"
        )
        assert 0 <= prediction.confidence_index <= 100, (
            f"confidence_index = {prediction.confidence_index} for {team_a} vs {team_b} "
            f"(style={style}). Expected [0, 100]."
        )
