"""
Property-based tests for recalibration and accuracy metrics.

Tests Properties 9 and 19 from the design document:
- Property 9: Recalibration adjustment magnitude bound
- Property 19: Accuracy metric calculation correctness

Validates: Requirements 4.4, 4.5, 4.7, 5.1, 5.2, 5.3
"""

import tempfile
from pathlib import Path

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.data.data_manager import DataManager, MatchResult, PredictionLogEntry
from src.engine.ensemble import EnsembleModel, EnsembleWeights
from src.tools.update_results import RecalibrationProcess
from src.utils.accuracy_tracker import AccuracyTracker


# ============================================================================
# Strategies
# ============================================================================


@st.composite
def valid_ensemble_weights(draw: st.DrawFn) -> EnsembleWeights:
    """Generate valid EnsembleWeights that sum to 1.0 and are in [0.10, 0.60].

    Strategy: generate 4 random values, normalize them, then clamp and re-normalize
    iteratively to ensure all constraints are met.
    """
    # Generate 4 positive values and normalize to sum=1.0
    raw = [
        draw(st.floats(min_value=0.1, max_value=1.0))
        for _ in range(4)
    ]
    total = sum(raw)
    normed = [v / total for v in raw]

    # Iteratively clamp to [0.10, 0.60] and re-normalize
    for _ in range(10):
        clamped = [max(0.10, min(0.60, v)) for v in normed]
        total = sum(clamped)
        normed = [v / total for v in clamped]

    return EnsembleWeights(
        poisson=normed[0],
        elo=normed[1],
        h2h=normed[2],
        dynamic=normed[3],
    )


# Strategy for performance dicts (each model gets a score 0.0 to 1.0)
performance_dicts = st.fixed_dictionaries({
    "poisson": st.floats(min_value=0.0, max_value=1.0),
    "elo": st.floats(min_value=0.0, max_value=1.0),
    "h2h": st.floats(min_value=0.0, max_value=1.0),
    "dynamic": st.floats(min_value=0.0, max_value=1.0),
})

# Strategy for score tuples (0-10 goals each)
score_tuples = st.tuples(
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
)

# Strategy for win/draw/lose probability tuples (non-negative, sum > 0)
wdl_tuples = st.tuples(
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
)

# Strategy for generating (prediction, actual) pairs for accuracy testing
prediction_actual_pairs = st.lists(
    st.tuples(
        # Predicted score
        score_tuples,
        # WDL probabilities
        wdl_tuples,
        # Actual score
        score_tuples,
    ),
    min_size=1,
    max_size=30,
)


# ============================================================================
# Property 9: Recalibration adjustment magnitude bound
# ============================================================================


class TestRecalibrationAdjustmentMagnitudeBound:
    """
    Property 9: Recalibration adjustment magnitude bound

    For any single recalibration event, the absolute change in each
    individual sub-model weight SHALL not exceed 0.05.

    **Validates: Requirements 4.4, 4.5**
    """

    @given(
        weights=valid_ensemble_weights(),
        performance=performance_dicts,
    )
    @settings(max_examples=200)
    def test_weight_adjustment_bounded_by_005(
        self, weights: EnsembleWeights, performance: dict[str, float]
    ) -> None:
        """
        For any valid starting weights and any performance dict,
        _adjust_weights SHALL not change any weight by more than 0.05.

        **Validates: Requirements 4.4**
        """
        assume(weights.validate())

        with tempfile.TemporaryDirectory() as tmp_dir:
            dm = DataManager(data_dir=Path(tmp_dir))
            ensemble = EnsembleModel(weights=weights)
            process = RecalibrationProcess(data_manager=dm, ensemble=ensemble)

            new_weights = process._adjust_weights(weights, performance)

            # Check each weight adjustment is bounded by 0.05
            assert abs(new_weights.poisson - weights.poisson) <= 0.05 + 1e-9, (
                f"Poisson weight changed by {abs(new_weights.poisson - weights.poisson):.6f}, "
                f"exceeds 0.05 bound"
            )
            assert abs(new_weights.elo - weights.elo) <= 0.05 + 1e-9, (
                f"Elo weight changed by {abs(new_weights.elo - weights.elo):.6f}, "
                f"exceeds 0.05 bound"
            )
            assert abs(new_weights.h2h - weights.h2h) <= 0.05 + 1e-9, (
                f"H2H weight changed by {abs(new_weights.h2h - weights.h2h):.6f}, "
                f"exceeds 0.05 bound"
            )
            assert abs(new_weights.dynamic - weights.dynamic) <= 0.05 + 1e-9, (
                f"Dynamic weight changed by {abs(new_weights.dynamic - weights.dynamic):.6f}, "
                f"exceeds 0.05 bound"
            )

    @given(
        weights=valid_ensemble_weights(),
        performance=performance_dicts,
    )
    @settings(max_examples=200)
    def test_adjusted_weights_remain_valid(
        self, weights: EnsembleWeights, performance: dict[str, float]
    ) -> None:
        """
        After recalibration, weights SHALL still be in [0.10, 0.60]
        and sum to 1.00.

        **Validates: Requirements 4.7**
        """
        assume(weights.validate())

        with tempfile.TemporaryDirectory() as tmp_dir:
            dm = DataManager(data_dir=Path(tmp_dir))
            ensemble = EnsembleModel(weights=weights)
            process = RecalibrationProcess(data_manager=dm, ensemble=ensemble)

            new_weights = process._adjust_weights(weights, performance)

            # Each weight in [0.10, 0.60]
            for name, val in [
                ("poisson", new_weights.poisson),
                ("elo", new_weights.elo),
                ("h2h", new_weights.h2h),
                ("dynamic", new_weights.dynamic),
            ]:
                assert 0.10 - 1e-9 <= val <= 0.60 + 1e-9, (
                    f"{name} weight {val:.6f} outside [0.10, 0.60]"
                )

            # Sum to 1.00
            total = (
                new_weights.poisson
                + new_weights.elo
                + new_weights.h2h
                + new_weights.dynamic
            )
            assert abs(total - 1.0) < 1e-6, (
                f"Weights sum to {total:.6f}, expected 1.0"
            )


# ============================================================================
# Property 19: Accuracy metric calculation correctness
# ============================================================================


class TestAccuracyMetricCalculationCorrectness:
    """
    Property 19: Accuracy metric calculation correctness

    For any non-empty list of (prediction, actual_result) pairs:
    - exact_score_rate SHALL equal count(predicted_score == actual_score) / total
    - direction_rate SHALL equal count(predicted_direction == actual_direction) / total
    - avg_goal_error SHALL equal sum(|predicted_total_goals − actual_total_goals|) / total

    **Validates: Requirements 5.1, 5.2, 5.3**
    """

    @given(pairs=prediction_actual_pairs)
    @settings(max_examples=200)
    def test_exact_score_rate_correctness(
        self, pairs: list[tuple[tuple[int, int], tuple[float, float, float], tuple[int, int]]]
    ) -> None:
        """
        exact_score_rate SHALL equal count(predicted_score == actual_score) / total * 100.

        **Validates: Requirements 5.1**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dm = DataManager(data_dir=Path(tmp_dir))
            tracker = AccuracyTracker(data_manager=dm)

            # Build paired list
            paired = self._build_paired_list(pairs)

            # Call the private method directly
            result = tracker._calc_exact_score_rate(paired)

            # Manually compute expected
            total = len(paired)
            hits = sum(
                1
                for pred, actual in paired
                if pred.predicted_score[0] == actual.score_a
                and pred.predicted_score[1] == actual.score_b
            )
            expected = round(hits / total * 100, 1)

            assert result == expected, (
                f"exact_score_rate={result}, expected={expected}"
            )

    @given(pairs=prediction_actual_pairs)
    @settings(max_examples=200)
    def test_direction_rate_correctness(
        self, pairs: list[tuple[tuple[int, int], tuple[float, float, float], tuple[int, int]]]
    ) -> None:
        """
        direction_rate SHALL equal count(predicted_direction == actual_direction) / total * 100.

        The AccuracyTracker determines predicted direction from the WDL probability
        tuple (highest probability wins), and actual direction from the score.

        **Validates: Requirements 5.2**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dm = DataManager(data_dir=Path(tmp_dir))
            tracker = AccuracyTracker(data_manager=dm)

            # Build paired list
            paired = self._build_paired_list(pairs)

            # Call the private method directly
            result = tracker._calc_direction_rate(paired)

            # Manually compute expected using the same logic as AccuracyTracker:
            # - predicted direction: from WDL tuple (max probability)
            # - actual direction: from score comparison
            total = len(paired)
            hits = 0
            for pred, actual in paired:
                # AccuracyTracker uses WDL probabilities for predicted direction
                pred_dir = self._get_predicted_direction_from_wdl(
                    pred.win_draw_lose
                )
                actual_dir = self._get_direction_from_score(
                    actual.score_a, actual.score_b
                )
                if pred_dir == actual_dir:
                    hits += 1

            expected = round(hits / total * 100, 1)

            assert result == expected, (
                f"direction_rate={result}, expected={expected}"
            )

    @given(pairs=prediction_actual_pairs)
    @settings(max_examples=200)
    def test_avg_goal_error_correctness(
        self, pairs: list[tuple[tuple[int, int], tuple[float, float, float], tuple[int, int]]]
    ) -> None:
        """
        avg_goal_error SHALL equal sum(|predicted_total_goals − actual_total_goals|) / total.

        **Validates: Requirements 5.3**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dm = DataManager(data_dir=Path(tmp_dir))
            tracker = AccuracyTracker(data_manager=dm)

            # Build paired list
            paired = self._build_paired_list(pairs)

            # Call the private method directly
            result = tracker._calc_avg_goal_error(paired)

            # Manually compute expected
            total = len(paired)
            total_error = sum(
                abs(
                    (pred.predicted_score[0] + pred.predicted_score[1])
                    - (actual.score_a + actual.score_b)
                )
                for pred, actual in paired
            )
            expected = round(total_error / total, 2)

            assert result == expected, (
                f"avg_goal_error={result}, expected={expected}"
            )

    # ========================================================================
    # Helpers
    # ========================================================================

    @staticmethod
    def _build_paired_list(
        pairs: list[tuple[tuple[int, int], tuple[float, float, float], tuple[int, int]]]
    ) -> list[tuple[PredictionLogEntry, MatchResult]]:
        """Build paired list of (PredictionLogEntry, MatchResult) from test data."""
        paired = []
        for i, (pred_score, wdl, actual_score) in enumerate(pairs):
            prediction = PredictionLogEntry(
                timestamp=f"2025-06-{i+1:02d}T12:00:00Z",
                match_id=f"MATCH-{i:04d}",
                team_a="TeamA",
                team_b="TeamB",
                predicted_score=pred_score,
                win_draw_lose=wdl,
                confidence_index=50,
                coach_style="分析師",
                model_weights={
                    "poisson": 0.40,
                    "elo": 0.25,
                    "h2h": 0.15,
                    "dynamic": 0.20,
                },
            )
            result = MatchResult(
                match_id=f"MATCH-{i:04d}",
                date=f"2025-06-{i+1:02d}",
                team_a="TeamA",
                team_b="TeamB",
                score_a=actual_score[0],
                score_b=actual_score[1],
                stage="group",
            )
            paired.append((prediction, result))
        return paired

    @staticmethod
    def _get_direction_from_score(score_a: int, score_b: int) -> str:
        """Determine direction from scores (mirrors AccuracyTracker logic)."""
        if score_a > score_b:
            return "win_a"
        elif score_a == score_b:
            return "draw"
        else:
            return "win_b"

    @staticmethod
    def _get_predicted_direction_from_wdl(wdl: tuple[float, float, float]) -> str:
        """Determine predicted direction from WDL probabilities (mirrors AccuracyTracker)."""
        win_prob, draw_prob, lose_prob = wdl
        if win_prob >= draw_prob and win_prob >= lose_prob:
            return "win_a"
        elif draw_prob >= win_prob and draw_prob >= lose_prob:
            return "draw"
        else:
            return "win_b"
