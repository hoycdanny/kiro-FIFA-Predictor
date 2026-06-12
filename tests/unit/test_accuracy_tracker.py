"""
Unit tests for the AccuracyTracker class.

Tests cover:
1. Zero matches returns "no data" message
2. < 3 matches returns "insufficient data" message
3. Exact score rate calculation correctness
4. Direction rate calculation correctness
5. Avg goal error calculation correctness
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.data_manager import DataManager, MatchResult, PredictionLogEntry
from src.utils.accuracy_tracker import AccuracyTracker, AccuracyReport, StyleAccuracy


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with empty data files."""
    # Empty match results
    match_results = {"matches": []}
    (tmp_path / "match_results.json").write_text(
        json.dumps(match_results), encoding="utf-8"
    )

    # Empty predictions log
    predictions_log = {"predictions": []}
    (tmp_path / "predictions_log.json").write_text(
        json.dumps(predictions_log), encoding="utf-8"
    )

    return tmp_path


@pytest.fixture
def data_manager(tmp_data_dir: Path) -> DataManager:
    """DataManager pointing at temporary data directory."""
    return DataManager(tmp_data_dir)


@pytest.fixture
def tracker(data_manager: DataManager) -> AccuracyTracker:
    """AccuracyTracker instance with empty data."""
    return AccuracyTracker(data_manager)


def _write_predictions(tmp_data_dir: Path, predictions: list[dict]) -> None:
    """Helper to write prediction log entries."""
    (tmp_data_dir / "predictions_log.json").write_text(
        json.dumps({"predictions": predictions}), encoding="utf-8"
    )


def _write_results(tmp_data_dir: Path, matches: list[dict]) -> None:
    """Helper to write match results."""
    (tmp_data_dir / "match_results.json").write_text(
        json.dumps({"matches": matches}), encoding="utf-8"
    )


def _make_prediction(
    match_id: str,
    team_a: str = "Brazil",
    team_b: str = "Argentina",
    predicted_score: list[int] = None,
    win_draw_lose: list[float] = None,
    confidence_index: int = 50,
    coach_style: str = "分析師",
) -> dict:
    """Create a prediction log entry dict."""
    return {
        "timestamp": "2026-06-10T14:30:00Z",
        "match_id": match_id,
        "team_a": team_a,
        "team_b": team_b,
        "predicted_score": predicted_score or [2, 1],
        "win_draw_lose": win_draw_lose or [0.55, 0.25, 0.20],
        "confidence_index": confidence_index,
        "coach_style": coach_style,
        "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
    }


def _make_result(
    match_id: str,
    team_a: str = "Brazil",
    team_b: str = "Argentina",
    score_a: int = 2,
    score_b: int = 1,
) -> dict:
    """Create a match result dict."""
    return {
        "match_id": match_id,
        "date": "2026-06-11",
        "team_a": team_a,
        "team_b": team_b,
        "score_a": score_a,
        "score_b": score_b,
        "stage": "group",
        "group": "C",
        "venue_country": "United States",
    }


# ============================================================================
# TEST: Zero matches returns "no data" message
# ============================================================================


class TestNoDataMessage:
    """Test that zero matched pairs returns the no data message."""

    def test_zero_matches_no_results(self, tracker: AccuracyTracker):
        """When no match results exist, return no data message."""
        report = tracker.calculate_report()
        assert isinstance(report, str)
        assert "目前無已完賽數據" in report

    def test_zero_matches_predictions_but_no_results(
        self, tracker: AccuracyTracker, tmp_data_dir: Path
    ):
        """When predictions exist but no results, return no data message."""
        _write_predictions(tmp_data_dir, [_make_prediction("GS-A-1")])
        report = tracker.calculate_report()
        assert isinstance(report, str)
        assert "目前無已完賽數據" in report

    def test_zero_matches_results_but_no_predictions(
        self, tracker: AccuracyTracker, tmp_data_dir: Path
    ):
        """When results exist but no predictions for those matches, return no data."""
        _write_results(tmp_data_dir, [_make_result("GS-A-1")])
        # No predictions for match GS-A-1
        report = tracker.calculate_report()
        assert isinstance(report, str)
        assert "目前無已完賽數據" in report


# ============================================================================
# TEST: < 3 matches returns "insufficient data" message
# ============================================================================


class TestInsufficientDataMessage:
    """Test that < 3 paired matches returns insufficient data message."""

    def test_one_match_insufficient(
        self, tracker: AccuracyTracker, tmp_data_dir: Path
    ):
        """One match pair returns insufficient data with count."""
        _write_predictions(tmp_data_dir, [_make_prediction("GS-A-1")])
        _write_results(tmp_data_dir, [_make_result("GS-A-1")])

        report = tracker.calculate_report()
        assert isinstance(report, str)
        assert "樣本數不足" in report
        assert "1 場" in report

    def test_two_matches_insufficient(
        self, tracker: AccuracyTracker, tmp_data_dir: Path
    ):
        """Two match pairs returns insufficient data with count."""
        _write_predictions(
            tmp_data_dir,
            [_make_prediction("GS-A-1"), _make_prediction("GS-A-2")],
        )
        _write_results(
            tmp_data_dir,
            [_make_result("GS-A-1"), _make_result("GS-A-2")],
        )

        report = tracker.calculate_report()
        assert isinstance(report, str)
        assert "樣本數不足" in report
        assert "2 場" in report

    def test_three_matches_sufficient(
        self, tracker: AccuracyTracker, tmp_data_dir: Path
    ):
        """Three match pairs returns a full AccuracyReport."""
        _write_predictions(
            tmp_data_dir,
            [
                _make_prediction("GS-A-1"),
                _make_prediction("GS-A-2"),
                _make_prediction("GS-A-3"),
            ],
        )
        _write_results(
            tmp_data_dir,
            [
                _make_result("GS-A-1"),
                _make_result("GS-A-2"),
                _make_result("GS-A-3"),
            ],
        )

        report = tracker.calculate_report()
        assert isinstance(report, AccuracyReport)
        assert report.total_matches == 3


# ============================================================================
# TEST: Exact score rate calculation
# ============================================================================


class TestExactScoreRate:
    """Test exact_score_rate = count(predicted == actual) / total × 100."""

    def test_all_exact_hits(self, tracker: AccuracyTracker, tmp_data_dir: Path):
        """All predictions exactly match results → 100.0%."""
        _write_predictions(
            tmp_data_dir,
            [
                _make_prediction("GS-A-1", predicted_score=[2, 1]),
                _make_prediction("GS-A-2", predicted_score=[1, 0]),
                _make_prediction("GS-A-3", predicted_score=[0, 0]),
            ],
        )
        _write_results(
            tmp_data_dir,
            [
                _make_result("GS-A-1", score_a=2, score_b=1),
                _make_result("GS-A-2", score_a=1, score_b=0),
                _make_result("GS-A-3", score_a=0, score_b=0),
            ],
        )

        report = tracker.calculate_report()
        assert isinstance(report, AccuracyReport)
        assert report.exact_score_rate == 100.0

    def test_no_exact_hits(self, tracker: AccuracyTracker, tmp_data_dir: Path):
        """No predictions match results exactly → 0.0%."""
        _write_predictions(
            tmp_data_dir,
            [
                _make_prediction("GS-A-1", predicted_score=[2, 1]),
                _make_prediction("GS-A-2", predicted_score=[1, 0]),
                _make_prediction("GS-A-3", predicted_score=[3, 0]),
            ],
        )
        _write_results(
            tmp_data_dir,
            [
                _make_result("GS-A-1", score_a=1, score_b=0),
                _make_result("GS-A-2", score_a=2, score_b=2),
                _make_result("GS-A-3", score_a=1, score_b=1),
            ],
        )

        report = tracker.calculate_report()
        assert isinstance(report, AccuracyReport)
        assert report.exact_score_rate == 0.0

    def test_partial_exact_hits(self, tracker: AccuracyTracker, tmp_data_dir: Path):
        """1 out of 3 predictions match → 33.3%."""
        _write_predictions(
            tmp_data_dir,
            [
                _make_prediction("GS-A-1", predicted_score=[2, 1]),  # hit
                _make_prediction("GS-A-2", predicted_score=[1, 0]),  # miss
                _make_prediction("GS-A-3", predicted_score=[3, 0]),  # miss
            ],
        )
        _write_results(
            tmp_data_dir,
            [
                _make_result("GS-A-1", score_a=2, score_b=1),
                _make_result("GS-A-2", score_a=2, score_b=2),
                _make_result("GS-A-3", score_a=1, score_b=0),
            ],
        )

        report = tracker.calculate_report()
        assert isinstance(report, AccuracyReport)
        assert report.exact_score_rate == 33.3


# ============================================================================
# TEST: Direction rate calculation
# ============================================================================


class TestDirectionRate:
    """Test direction_rate = count(predicted_direction == actual_direction) / total × 100."""

    def test_all_direction_hits(self, tracker: AccuracyTracker, tmp_data_dir: Path):
        """All direction predictions correct → 100.0%."""
        # Predict win_a (win_prob highest), actual win_a
        _write_predictions(
            tmp_data_dir,
            [
                _make_prediction("GS-A-1", win_draw_lose=[0.60, 0.25, 0.15]),  # win_a
                _make_prediction("GS-A-2", win_draw_lose=[0.20, 0.50, 0.30]),  # draw
                _make_prediction("GS-A-3", win_draw_lose=[0.15, 0.25, 0.60]),  # win_b
            ],
        )
        _write_results(
            tmp_data_dir,
            [
                _make_result("GS-A-1", score_a=2, score_b=1),  # win_a
                _make_result("GS-A-2", score_a=1, score_b=1),  # draw
                _make_result("GS-A-3", score_a=0, score_b=2),  # win_b
            ],
        )

        report = tracker.calculate_report()
        assert isinstance(report, AccuracyReport)
        assert report.direction_rate == 100.0

    def test_no_direction_hits(self, tracker: AccuracyTracker, tmp_data_dir: Path):
        """No direction predictions correct → 0.0%."""
        _write_predictions(
            tmp_data_dir,
            [
                _make_prediction("GS-A-1", win_draw_lose=[0.60, 0.25, 0.15]),  # win_a
                _make_prediction("GS-A-2", win_draw_lose=[0.20, 0.50, 0.30]),  # draw
                _make_prediction("GS-A-3", win_draw_lose=[0.15, 0.25, 0.60]),  # win_b
            ],
        )
        _write_results(
            tmp_data_dir,
            [
                _make_result("GS-A-1", score_a=0, score_b=2),  # win_b (not win_a)
                _make_result("GS-A-2", score_a=3, score_b=0),  # win_a (not draw)
                _make_result("GS-A-3", score_a=2, score_b=0),  # win_a (not win_b)
            ],
        )

        report = tracker.calculate_report()
        assert isinstance(report, AccuracyReport)
        assert report.direction_rate == 0.0

    def test_partial_direction_hits(self, tracker: AccuracyTracker, tmp_data_dir: Path):
        """2 out of 3 direction predictions correct → 66.7%."""
        _write_predictions(
            tmp_data_dir,
            [
                _make_prediction("GS-A-1", win_draw_lose=[0.60, 0.25, 0.15]),  # win_a
                _make_prediction("GS-A-2", win_draw_lose=[0.20, 0.50, 0.30]),  # draw
                _make_prediction("GS-A-3", win_draw_lose=[0.15, 0.25, 0.60]),  # win_b
            ],
        )
        _write_results(
            tmp_data_dir,
            [
                _make_result("GS-A-1", score_a=1, score_b=0),  # win_a ✓
                _make_result("GS-A-2", score_a=0, score_b=0),  # draw ✓
                _make_result("GS-A-3", score_a=2, score_b=0),  # win_a ✗
            ],
        )

        report = tracker.calculate_report()
        assert isinstance(report, AccuracyReport)
        assert report.direction_rate == 66.7


# ============================================================================
# TEST: Average goal error calculation
# ============================================================================


class TestAvgGoalError:
    """Test avg_goal_error = sum(|predicted_total - actual_total|) / total."""

    def test_zero_error(self, tracker: AccuracyTracker, tmp_data_dir: Path):
        """Predicted total goals match actual total → 0.0 error."""
        _write_predictions(
            tmp_data_dir,
            [
                _make_prediction("GS-A-1", predicted_score=[2, 1]),  # total = 3
                _make_prediction("GS-A-2", predicted_score=[1, 1]),  # total = 2
                _make_prediction("GS-A-3", predicted_score=[0, 0]),  # total = 0
            ],
        )
        _write_results(
            tmp_data_dir,
            [
                _make_result("GS-A-1", score_a=2, score_b=1),  # total = 3
                _make_result("GS-A-2", score_a=0, score_b=2),  # total = 2
                _make_result("GS-A-3", score_a=0, score_b=0),  # total = 0
            ],
        )

        report = tracker.calculate_report()
        assert isinstance(report, AccuracyReport)
        assert report.avg_goal_error == 0.0

    def test_known_error(self, tracker: AccuracyTracker, tmp_data_dir: Path):
        """Known goal errors: |3-2| + |2-4| + |0-1| = 1+2+1 = 4, avg = 4/3 = 1.33."""
        _write_predictions(
            tmp_data_dir,
            [
                _make_prediction("GS-A-1", predicted_score=[2, 1]),  # total = 3
                _make_prediction("GS-A-2", predicted_score=[1, 1]),  # total = 2
                _make_prediction("GS-A-3", predicted_score=[0, 0]),  # total = 0
            ],
        )
        _write_results(
            tmp_data_dir,
            [
                _make_result("GS-A-1", score_a=1, score_b=1),  # total = 2, error = 1
                _make_result("GS-A-2", score_a=2, score_b=2),  # total = 4, error = 2
                _make_result("GS-A-3", score_a=1, score_b=0),  # total = 1, error = 1
            ],
        )

        report = tracker.calculate_report()
        assert isinstance(report, AccuracyReport)
        assert report.avg_goal_error == 1.33

    def test_large_error(self, tracker: AccuracyTracker, tmp_data_dir: Path):
        """Large errors: |1-5| + |1-0| + |1-6| = 4+1+5 = 10, avg = 10/3 = 3.33."""
        _write_predictions(
            tmp_data_dir,
            [
                _make_prediction("GS-A-1", predicted_score=[1, 0]),  # total = 1
                _make_prediction("GS-A-2", predicted_score=[0, 1]),  # total = 1
                _make_prediction("GS-A-3", predicted_score=[1, 0]),  # total = 1
            ],
        )
        _write_results(
            tmp_data_dir,
            [
                _make_result("GS-A-1", score_a=3, score_b=2),  # total = 5, error = 4
                _make_result("GS-A-2", score_a=0, score_b=0),  # total = 0, error = 1
                _make_result("GS-A-3", score_a=4, score_b=2),  # total = 6, error = 5
            ],
        )

        report = tracker.calculate_report()
        assert isinstance(report, AccuracyReport)
        assert report.avg_goal_error == 3.33
