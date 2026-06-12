"""
Unit tests for the Recalibration Process.

Tests verify:
1. _adjust_weights respects ±0.05 bound
2. _adjust_weights maintains sum = 1.0
3. Dynamic factor updates (streak counting)
4. Report generation with sample data
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.data_manager import DataManager, MatchResult, TeamProfile
from src.engine.ensemble import EnsembleModel, EnsembleWeights
from src.tools.update_results import RecalibrationProcess, RecalibrationReport


# ============================================================================
# HELPERS
# ============================================================================


def _make_team(
    name: str = "TeamA",
    confederation: str = "UEFA",
    win_streak: int = 0,
    loss_streak: int = 0,
    last_match_date: str | None = None,
) -> TeamProfile:
    """Create a minimal TeamProfile for testing."""
    return TeamProfile(
        name=name,
        name_zh="",
        aliases=[],
        confederation=confederation,
        fifa_ranking=1,
        fifa_points=1500.0,
        elo_rating=1800,
        group="A",
        recent_goals_avg=1.5,
        recent_conceded_avg=1.0,
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
    )


def _make_data_manager(tmp_path: Path) -> DataManager:
    """Create a DataManager with minimal test data files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Create empty match_results.json
    (data_dir / "match_results.json").write_text(
        json.dumps({"matches": []}), encoding="utf-8"
    )
    # Create empty predictions_log.json
    (data_dir / "predictions_log.json").write_text(
        json.dumps({"predictions": []}), encoding="utf-8"
    )
    # Create calibration.json
    (data_dir / "calibration.json").write_text(
        json.dumps({
            "current_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
            "weight_history": [],
            "accuracy_records": [],
        }),
        encoding="utf-8",
    )
    # Create teams.json with two teams
    teams = [
        {
            "name": "Brazil",
            "name_zh": "巴西",
            "aliases": ["BRA"],
            "confederation": "CONMEBOL",
            "fifa_ranking": 1,
            "fifa_points": 1840.0,
            "elo_rating": 2150,
            "group": "A",
            "recent_goals_avg": 2.0,
            "recent_conceded_avg": 0.8,
            "recent_win_rate": 70.0,
            "recent_draw_rate": 20.0,
            "recent_loss_rate": 10.0,
            "neutral_win_rate": 60.0,
            "best_wc_result": "Champion",
            "vs_top20_win_rate": 50.0,
            "wc_first_match_win_rate": 80.0,
            "penalty_shootout_win_rate": 60.0,
            "first_half_goal_pct": 45.0,
            "second_half_goal_pct": 55.0,
            "clean_sheet_rate": 40.0,
            "failed_to_score_rate": 10.0,
            "current_win_streak": 2,
            "current_loss_streak": 0,
            "last_match_date": "2026-06-10",
            "eliminated_by_2022": None,
        },
        {
            "name": "Argentina",
            "name_zh": "阿根廷",
            "aliases": ["ARG"],
            "confederation": "CONMEBOL",
            "fifa_ranking": 2,
            "fifa_points": 1820.0,
            "elo_rating": 2100,
            "group": "B",
            "recent_goals_avg": 1.8,
            "recent_conceded_avg": 0.9,
            "recent_win_rate": 65.0,
            "recent_draw_rate": 20.0,
            "recent_loss_rate": 15.0,
            "neutral_win_rate": 55.0,
            "best_wc_result": "Champion",
            "vs_top20_win_rate": 45.0,
            "wc_first_match_win_rate": 70.0,
            "penalty_shootout_win_rate": 55.0,
            "first_half_goal_pct": 40.0,
            "second_half_goal_pct": 60.0,
            "clean_sheet_rate": 35.0,
            "failed_to_score_rate": 12.0,
            "current_win_streak": 0,
            "current_loss_streak": 1,
            "last_match_date": "2026-06-09",
            "eliminated_by_2022": None,
        },
    ]
    (data_dir / "teams.json").write_text(
        json.dumps({"teams": teams}, ensure_ascii=False), encoding="utf-8"
    )

    return DataManager(data_dir)


def _make_process(tmp_path: Path) -> RecalibrationProcess:
    """Create a RecalibrationProcess with test dependencies."""
    dm = _make_data_manager(tmp_path)
    weights = EnsembleWeights(poisson=0.40, elo=0.25, h2h=0.15, dynamic=0.20)
    ensemble = MagicMock(spec=EnsembleModel)
    ensemble.weights = weights
    ensemble.update_weights = MagicMock(side_effect=lambda w: setattr(ensemble, "weights", w))
    return RecalibrationProcess(data_manager=dm, ensemble=ensemble)


# ============================================================================
# TEST: _adjust_weights respects ±0.05 bound
# ============================================================================


class TestAdjustWeightsBounds:
    """Tests that _adjust_weights respects the ±0.05 maximum adjustment."""

    def test_adjustment_capped_at_plus_005(self, tmp_path):
        """When performance strongly favors a model, adjustment is capped at +0.05."""
        process = _make_process(tmp_path)
        current = EnsembleWeights(poisson=0.40, elo=0.25, h2h=0.15, dynamic=0.20)

        # Give poisson extremely high performance to try to force large adjustment
        performance = {"poisson": 1.0, "elo": 0.0, "h2h": 0.0, "dynamic": 0.0}
        result = process._adjust_weights(current, performance)

        # Each individual weight change should be at most 0.05
        assert result.poisson - current.poisson <= 0.05 + 1e-9
        assert current.elo - result.elo <= 0.05 + 1e-9
        assert current.h2h - result.h2h <= 0.05 + 1e-9
        assert current.dynamic - result.dynamic <= 0.05 + 1e-9

    def test_adjustment_capped_at_minus_005(self, tmp_path):
        """When performance strongly disfavors a model, adjustment is capped at -0.05."""
        process = _make_process(tmp_path)
        current = EnsembleWeights(poisson=0.40, elo=0.25, h2h=0.15, dynamic=0.20)

        # Give poisson extremely low performance
        performance = {"poisson": 0.0, "elo": 1.0, "h2h": 1.0, "dynamic": 1.0}
        result = process._adjust_weights(current, performance)

        # Poisson should decrease by at most 0.05
        assert current.poisson - result.poisson <= 0.05 + 1e-9

    def test_weights_stay_in_valid_range(self, tmp_path):
        """All weights remain in [0.10, 0.60] after adjustment."""
        process = _make_process(tmp_path)
        # Start near boundaries
        current = EnsembleWeights(poisson=0.55, elo=0.15, h2h=0.15, dynamic=0.15)

        performance = {"poisson": 1.0, "elo": 0.0, "h2h": 0.5, "dynamic": 0.5}
        result = process._adjust_weights(current, performance)

        assert 0.10 <= result.poisson <= 0.60
        assert 0.10 <= result.elo <= 0.60
        assert 0.10 <= result.h2h <= 0.60
        assert 0.10 <= result.dynamic <= 0.60

    def test_no_adjustment_with_equal_performance(self, tmp_path):
        """When all models perform equally, weights should not change significantly."""
        process = _make_process(tmp_path)
        current = EnsembleWeights(poisson=0.40, elo=0.25, h2h=0.15, dynamic=0.20)

        performance = {"poisson": 0.5, "elo": 0.5, "h2h": 0.5, "dynamic": 0.5}
        result = process._adjust_weights(current, performance)

        # Weights should remain essentially unchanged (within tolerance from normalization)
        assert abs(result.poisson - current.poisson) < 0.01
        assert abs(result.elo - current.elo) < 0.01
        assert abs(result.h2h - current.h2h) < 0.01
        assert abs(result.dynamic - current.dynamic) < 0.01


# ============================================================================
# TEST: _adjust_weights maintains sum = 1.0
# ============================================================================


class TestAdjustWeightsSum:
    """Tests that _adjust_weights always produces weights summing to 1.0."""

    def test_sum_equals_one_after_adjustment(self, tmp_path):
        """Adjusted weights should sum to 1.00."""
        process = _make_process(tmp_path)
        current = EnsembleWeights(poisson=0.40, elo=0.25, h2h=0.15, dynamic=0.20)

        performance = {"poisson": 0.8, "elo": 0.3, "h2h": 0.6, "dynamic": 0.4}
        result = process._adjust_weights(current, performance)

        total = result.poisson + result.elo + result.h2h + result.dynamic
        assert abs(total - 1.0) < 1e-9

    def test_sum_equals_one_extreme_performance(self, tmp_path):
        """Sum = 1.0 even with extreme performance differences."""
        process = _make_process(tmp_path)
        current = EnsembleWeights(poisson=0.40, elo=0.25, h2h=0.15, dynamic=0.20)

        performance = {"poisson": 1.0, "elo": 0.0, "h2h": 0.0, "dynamic": 0.0}
        result = process._adjust_weights(current, performance)

        total = result.poisson + result.elo + result.h2h + result.dynamic
        assert abs(total - 1.0) < 1e-9

    def test_sum_equals_one_from_near_boundaries(self, tmp_path):
        """Sum = 1.0 when starting from weights near min/max bounds."""
        process = _make_process(tmp_path)
        current = EnsembleWeights(poisson=0.55, elo=0.10, h2h=0.15, dynamic=0.20)

        performance = {"poisson": 0.9, "elo": 0.1, "h2h": 0.5, "dynamic": 0.5}
        result = process._adjust_weights(current, performance)

        total = result.poisson + result.elo + result.h2h + result.dynamic
        assert abs(total - 1.0) < 1e-9

    def test_validates_after_adjustment(self, tmp_path):
        """Result should pass EnsembleWeights.validate()."""
        process = _make_process(tmp_path)
        current = EnsembleWeights(poisson=0.40, elo=0.25, h2h=0.15, dynamic=0.20)

        performance = {"poisson": 0.7, "elo": 0.5, "h2h": 0.3, "dynamic": 0.6}
        result = process._adjust_weights(current, performance)

        assert result.validate()


# ============================================================================
# TEST: Dynamic factor updates (streak counting)
# ============================================================================


class TestDynamicFactorUpdates:
    """Tests for team streak updates after match results."""

    def test_win_increments_win_streak(self, tmp_path):
        """Winning team's win_streak should increment, loss_streak resets."""
        process = _make_process(tmp_path)
        team = _make_team(name="Brazil", win_streak=2, loss_streak=0)

        process._update_team_streak(team, goals_for=2, goals_against=1, match_date="2026-06-15")

        assert team.current_win_streak == 3
        assert team.current_loss_streak == 0
        assert team.last_match_date == "2026-06-15"

    def test_loss_increments_loss_streak(self, tmp_path):
        """Losing team's loss_streak should increment, win_streak resets."""
        process = _make_process(tmp_path)
        team = _make_team(name="Brazil", win_streak=5, loss_streak=0)

        process._update_team_streak(team, goals_for=0, goals_against=2, match_date="2026-06-15")

        assert team.current_win_streak == 0
        assert team.current_loss_streak == 1
        assert team.last_match_date == "2026-06-15"

    def test_draw_resets_both_streaks(self, tmp_path):
        """Draw should reset both win_streak and loss_streak to 0."""
        process = _make_process(tmp_path)
        team = _make_team(name="Brazil", win_streak=3, loss_streak=0)

        process._update_team_streak(team, goals_for=1, goals_against=1, match_date="2026-06-15")

        assert team.current_win_streak == 0
        assert team.current_loss_streak == 0
        assert team.last_match_date == "2026-06-15"

    def test_update_dynamic_factors_multiple_results(self, tmp_path):
        """Multiple results should update both teams correctly."""
        process = _make_process(tmp_path)
        teams = [
            _make_team(name="Brazil", win_streak=0, loss_streak=0),
            _make_team(name="Argentina", win_streak=0, loss_streak=0),
        ]

        results = [
            MatchResult(
                match_id="GS-A-1",
                date="2026-06-15",
                team_a="Brazil",
                team_b="Argentina",
                score_a=2,
                score_b=1,
                stage="group",
                group="A",
            )
        ]

        updated = process._update_dynamic_factors(teams, results)
        team_map = {t.name: t for t in updated}

        # Brazil won
        assert team_map["Brazil"].current_win_streak == 1
        assert team_map["Brazil"].current_loss_streak == 0
        # Argentina lost
        assert team_map["Argentina"].current_win_streak == 0
        assert team_map["Argentina"].current_loss_streak == 1

    def test_sequential_wins_build_streak(self, tmp_path):
        """Multiple consecutive wins should build up the win streak."""
        process = _make_process(tmp_path)
        team = _make_team(name="Brazil", win_streak=0, loss_streak=0)

        # 3 consecutive wins
        process._update_team_streak(team, 2, 0, "2026-06-11")
        assert team.current_win_streak == 1

        process._update_team_streak(team, 1, 0, "2026-06-14")
        assert team.current_win_streak == 2

        process._update_team_streak(team, 3, 1, "2026-06-17")
        assert team.current_win_streak == 3
        assert team.current_loss_streak == 0

    def test_loss_after_streak_resets(self, tmp_path):
        """A loss after a win streak resets win_streak and starts loss_streak."""
        process = _make_process(tmp_path)
        team = _make_team(name="Brazil", win_streak=5, loss_streak=0)

        process._update_team_streak(team, 0, 1, "2026-06-20")
        assert team.current_win_streak == 0
        assert team.current_loss_streak == 1


# ============================================================================
# TEST: Report generation with sample data
# ============================================================================


class TestReportGeneration:
    """Tests for recalibration report generation."""

    def test_empty_results_returns_zero_report(self, tmp_path):
        """No results should produce an empty report with no changes."""
        process = _make_process(tmp_path)

        report = asyncio.run(process.update_results(match_results=[]))

        assert report.matches_evaluated == 0
        assert report.weights_before == report.weights_after
        assert report.exact_score_hits == 0
        assert report.direction_hits == 0
        assert report.avg_goal_error == 0.0
        assert report.systematic_bias is None

    def test_report_with_matching_predictions(self, tmp_path):
        """Report correctly counts hits when predictions exist in log."""
        process = _make_process(tmp_path)

        # Write a prediction to the log
        predictions = {
            "predictions": [
                {
                    "timestamp": "2026-06-10T14:30:00Z",
                    "match_id": "GS-A-1",
                    "team_a": "Brazil",
                    "team_b": "Argentina",
                    "predicted_score": [2, 1],
                    "win_draw_lose": [0.60, 0.25, 0.15],
                    "confidence_index": 70,
                    "coach_style": "分析師",
                    "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
                }
            ]
        }
        pred_path = process.data_manager.data_dir / "predictions_log.json"
        pred_path.write_text(json.dumps(predictions), encoding="utf-8")

        # Provide a matching result (exact score hit!)
        results = [
            MatchResult(
                match_id="GS-A-1",
                date="2026-06-15",
                team_a="Brazil",
                team_b="Argentina",
                score_a=2,
                score_b=1,
                stage="group",
                group="A",
            )
        ]

        report = asyncio.run(process.update_results(match_results=results))

        assert report.matches_evaluated == 1
        assert report.exact_score_hits == 1
        assert report.direction_hits == 1
        assert report.avg_goal_error == 0.0

    def test_report_direction_hit_no_exact(self, tmp_path):
        """Direction hit when predicted winner is correct but score is off."""
        process = _make_process(tmp_path)

        predictions = {
            "predictions": [
                {
                    "timestamp": "2026-06-10T14:30:00Z",
                    "match_id": "GS-A-1",
                    "team_a": "Brazil",
                    "team_b": "Argentina",
                    "predicted_score": [2, 0],  # Predicted 2-0
                    "win_draw_lose": [0.60, 0.25, 0.15],
                    "confidence_index": 70,
                    "coach_style": "分析師",
                    "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
                }
            ]
        }
        pred_path = process.data_manager.data_dir / "predictions_log.json"
        pred_path.write_text(json.dumps(predictions), encoding="utf-8")

        # Actual: 3-1 (direction correct, score wrong)
        results = [
            MatchResult(
                match_id="GS-A-1",
                date="2026-06-15",
                team_a="Brazil",
                team_b="Argentina",
                score_a=3,
                score_b=1,
                stage="group",
                group="A",
            )
        ]

        report = asyncio.run(process.update_results(match_results=results))

        assert report.matches_evaluated == 1
        assert report.exact_score_hits == 0
        assert report.direction_hits == 1
        # Predicted total: 2, actual total: 4, error = 2
        assert report.avg_goal_error == 2.0

    def test_report_no_predictions_for_result(self, tmp_path):
        """Results with no matching predictions should not count."""
        process = _make_process(tmp_path)

        # Empty predictions log
        results = [
            MatchResult(
                match_id="GS-X-1",
                date="2026-06-15",
                team_a="Brazil",
                team_b="Argentina",
                score_a=1,
                score_b=0,
                stage="group",
                group="A",
            )
        ]

        report = asyncio.run(process.update_results(match_results=results))

        # No comparisons possible since no matching predictions
        assert report.matches_evaluated == 0

    def test_systematic_bias_triggered(self, tmp_path):
        """Systematic bias report is generated when direction < 50% after 5+ matches."""
        process = _make_process(tmp_path)

        # Create 5 predictions that are all wrong in direction
        predictions = {"predictions": []}
        results = []
        for i in range(5):
            predictions["predictions"].append({
                "timestamp": f"2026-06-{10 + i}T14:30:00Z",
                "match_id": f"GS-A-{i + 1}",
                "team_a": "Brazil",
                "team_b": "Argentina",
                "predicted_score": [2, 0],  # Predict Brazil wins
                "win_draw_lose": [0.70, 0.20, 0.10],
                "confidence_index": 70,
                "coach_style": "分析師",
                "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
            })
            results.append(
                MatchResult(
                    match_id=f"GS-A-{i + 1}",
                    date=f"2026-06-{15 + i}",
                    team_a="Brazil",
                    team_b="Argentina",
                    score_a=0,
                    score_b=1,  # Argentina wins (direction wrong!)
                    stage="group",
                    group="A",
                )
            )

        pred_path = process.data_manager.data_dir / "predictions_log.json"
        pred_path.write_text(json.dumps(predictions), encoding="utf-8")

        report = asyncio.run(process.update_results(match_results=results))

        assert report.matches_evaluated == 5
        assert report.direction_hits == 0
        assert report.systematic_bias is not None
        assert "跨聯盟" in report.systematic_bias
