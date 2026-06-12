"""
Integration tests for edge cases and error handling.

Tests:
- Invalid team name returns ≤ 3 suggestions (Req 1.5)
- Invalid group ID returns all valid options (Req 2.4)
- Invalid coach style returns three valid options (Req 1.7)
- Sub-model failure triggers graceful degradation (Req 7.8)
- Accuracy report with 0 matches and < 3 matches (Req 5.7, 5.8)
- Recalibration systematic bias report trigger (Req 4.8)

Requirements: 1.5, 1.7, 2.4, 5.7, 5.8, 7.8, 4.8
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.data.data_manager import (
    DataManager,
    MatchResult,
    PredictionError,
    PredictionLogEntry,
    TeamProfile,
)
from src.engine.ensemble import AllModelsFailedError, EnsembleModel, EnsembleWeights
from src.engine.dixon_coles import DixonColesModel
from src.engine.elo_model import EloModel
from src.engine.h2h_model import H2HModel
from src.engine.dynamic_factor import DynamicFactorModel
from src.utils.accuracy_tracker import AccuracyTracker
from src.utils.validator import CoachStyleType, InputValidator
from src.tools.update_results import RecalibrationProcess


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def validator() -> InputValidator:
    """Create an InputValidator with default TeamMatcher."""
    return InputValidator()


def _make_team(name: str, confederation: str = "UEFA", elo: int = 1800) -> TeamProfile:
    """Helper to create a minimal TeamProfile for testing."""
    return TeamProfile(
        name=name,
        name_zh="測試隊",
        aliases=[],
        confederation=confederation,
        fifa_ranking=10,
        fifa_points=1500.0,
        elo_rating=elo,
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
        failed_to_score_rate=20.0,
        current_win_streak=0,
        current_loss_streak=0,
        last_match_date=None,
        eliminated_by_2022=None,
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with minimal required files."""
    # Create empty match_results.json
    (tmp_path / "match_results.json").write_text(
        json.dumps({"matches": []}), encoding="utf-8"
    )
    # Create empty predictions_log.json
    (tmp_path / "predictions_log.json").write_text(
        json.dumps({"predictions": []}), encoding="utf-8"
    )
    # Create calibration.json
    (tmp_path / "calibration.json").write_text(
        json.dumps({
            "current_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
            "weight_history": [],
            "accuracy_records": [],
        }),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def data_manager(data_dir: Path) -> DataManager:
    """Create a DataManager pointing to the temp directory."""
    return DataManager(data_dir)


# ============================================================================
# TEST: Invalid team name returns ≤ 3 suggestions (Req 1.5)
# ============================================================================


class TestInvalidTeamSuggestions:
    """Invalid team name should return PredictionError with ≤ 3 suggestions."""

    def test_completely_invalid_team_max_3_suggestions(self, validator: InputValidator) -> None:
        """A totally made-up name should return at most 3 suggestions."""
        result = validator.validate_team("Xylophones FC")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_TEAM"
        assert len(result.suggestions) <= 3

    def test_slightly_misspelled_team_max_3_suggestions(self, validator: InputValidator) -> None:
        """A clearly invalid name still returns at most 3 suggestions."""
        result = validator.validate_team("Brzzzzl")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_TEAM"
        assert len(result.suggestions) <= 3
        assert len(result.suggestions) > 0  # Should have some suggestions

    def test_numeric_input_max_3_suggestions(self, validator: InputValidator) -> None:
        """Numeric input produces at most 3 suggestions."""
        result = validator.validate_team("12345")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_TEAM"
        assert len(result.suggestions) <= 3

    def test_single_char_invalid_team(self, validator: InputValidator) -> None:
        """Single character non-matching input returns ≤ 3 suggestions."""
        result = validator.validate_team("Z")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_TEAM"
        assert len(result.suggestions) <= 3

    def test_suggestions_are_real_team_names(self, validator: InputValidator) -> None:
        """Suggestions should be actual team names from the 48-team list."""
        from src.utils.constants import ALL_TEAMS

        result = validator.validate_team("Unknownistan")  # no real match
        assert isinstance(result, PredictionError)
        for suggestion in result.suggestions:
            assert suggestion in ALL_TEAMS


# ============================================================================
# TEST: Invalid group ID returns all valid options (Req 2.4)
# ============================================================================


class TestInvalidGroupReturnsAllOptions:
    """Invalid group ID should return PredictionError listing all valid groups A-L."""

    def test_invalid_group_letter_lists_all(self, validator: InputValidator) -> None:
        """Group ID 'M' (out of range) returns all valid options A-L."""
        result = validator.validate_group("M")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_GROUP"
        assert sorted(result.suggestions) == list("ABCDEFGHIJKL")

    def test_invalid_group_number_lists_all(self, validator: InputValidator) -> None:
        """Numeric group ID returns all valid options A-L."""
        result = validator.validate_group("1")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_GROUP"
        assert sorted(result.suggestions) == list("ABCDEFGHIJKL")

    def test_invalid_group_special_char_lists_all(self, validator: InputValidator) -> None:
        """Special character as group ID returns all valid options."""
        result = validator.validate_group("!")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_GROUP"
        assert sorted(result.suggestions) == list("ABCDEFGHIJKL")

    def test_invalid_group_empty_lists_all(self, validator: InputValidator) -> None:
        """Empty group ID returns all valid options."""
        result = validator.validate_group("")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_GROUP"
        assert sorted(result.suggestions) == list("ABCDEFGHIJKL")


# ============================================================================
# TEST: Invalid coach style returns three valid options (Req 1.7)
# ============================================================================


class TestInvalidCoachStyleReturnsOptions:
    """Invalid coach style should return PredictionError with 3 valid options."""

    def test_random_word_returns_3_options(self, validator: InputValidator) -> None:
        """Random word should return exactly 3 valid style suggestions."""
        result = validator.validate_coach_style("random_style")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_STYLE"
        assert len(result.suggestions) == 3

    def test_empty_string_returns_3_options(self, validator: InputValidator) -> None:
        """Empty input should return exactly 3 valid style suggestions."""
        result = validator.validate_coach_style("")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_STYLE"
        assert len(result.suggestions) == 3

    def test_numeric_returns_3_options(self, validator: InputValidator) -> None:
        """Numeric input should return 3 valid style suggestions."""
        result = validator.validate_coach_style("123")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_STYLE"
        assert len(result.suggestions) == 3

    def test_suggestions_contain_all_styles(self, validator: InputValidator) -> None:
        """The 3 suggestions should reference all 3 valid styles."""
        result = validator.validate_coach_style("invalid")
        assert isinstance(result, PredictionError)
        # Each suggestion references one style
        combined = " ".join(result.suggestions)
        assert "分析師" in combined or "analyst" in combined
        assert "反向思考者" in combined or "contrarian" in combined
        assert "戰術家" in combined or "tactician" in combined


# ============================================================================
# TEST: Sub-model failure triggers graceful degradation (Req 7.8)
# ============================================================================


class TestSubModelFailureGracefulDegradation:
    """When a sub-model throws, EnsembleModel should fall back gracefully."""

    def test_single_model_failure_still_predicts(self) -> None:
        """If one sub-model fails, prediction still succeeds with remaining models."""
        team_a = _make_team("Brazil", "CONMEBOL", 2100)
        team_b = _make_team("Argentina", "CONMEBOL", 2050)

        # Create ensemble with a broken Dixon-Coles model
        broken_dc = DixonColesModel()
        broken_dc.predict = MagicMock(side_effect=RuntimeError("Model crashed"))

        ensemble = EnsembleModel(
            weights=EnsembleWeights(),
            dixon_coles=broken_dc,
            elo_model=EloModel(),
            h2h_model=H2HModel(),
            dynamic_factor=DynamicFactorModel(),
        )

        result = ensemble.predict_with_fallback(team_a, team_b)

        # Should still return valid probabilities
        win_a, draw, win_b = result
        assert abs(win_a + draw + win_b - 1.0) < 1e-6
        assert win_a >= 0 and draw >= 0 and win_b >= 0

    def test_two_models_failure_still_predicts(self) -> None:
        """If two sub-models fail, prediction still succeeds with remaining."""
        team_a = _make_team("Germany", "UEFA", 1900)
        team_b = _make_team("Japan", "AFC", 1750)

        broken_dc = DixonColesModel()
        broken_dc.predict = MagicMock(side_effect=ValueError("NaN produced"))

        broken_elo = EloModel()
        broken_elo.predict = MagicMock(side_effect=ZeroDivisionError("division by zero"))

        ensemble = EnsembleModel(
            weights=EnsembleWeights(),
            dixon_coles=broken_dc,
            elo_model=broken_elo,
            h2h_model=H2HModel(),
            dynamic_factor=DynamicFactorModel(),
        )

        result = ensemble.predict_with_fallback(team_a, team_b)

        win_a, draw, win_b = result
        assert abs(win_a + draw + win_b - 1.0) < 1e-6
        assert win_a >= 0 and draw >= 0 and win_b >= 0

    def test_all_models_failure_raises_error(self) -> None:
        """If ALL sub-models fail, AllModelsFailedError is raised."""
        team_a = _make_team("France", "UEFA", 2000)
        team_b = _make_team("Spain", "UEFA", 1950)

        broken_dc = DixonColesModel()
        broken_dc.predict = MagicMock(side_effect=RuntimeError("fail"))
        broken_elo = EloModel()
        broken_elo.predict = MagicMock(side_effect=RuntimeError("fail"))
        broken_h2h = H2HModel()
        broken_h2h.predict = MagicMock(side_effect=RuntimeError("fail"))
        broken_dyn = DynamicFactorModel()
        broken_dyn.calculate_adjustment = MagicMock(side_effect=RuntimeError("fail"))

        ensemble = EnsembleModel(
            weights=EnsembleWeights(),
            dixon_coles=broken_dc,
            elo_model=broken_elo,
            h2h_model=broken_h2h,
            dynamic_factor=broken_dyn,
        )

        with pytest.raises(AllModelsFailedError):
            ensemble.predict_with_fallback(team_a, team_b)

    def test_probabilities_sum_to_one_after_degradation(self) -> None:
        """After graceful degradation, probabilities still sum to 1.0."""
        team_a = _make_team("England", "UEFA", 1950)
        team_b = _make_team("Netherlands", "UEFA", 1900)

        broken_h2h = H2HModel()
        broken_h2h.predict = MagicMock(side_effect=Exception("H2H data corrupted"))

        ensemble = EnsembleModel(
            weights=EnsembleWeights(),
            dixon_coles=DixonColesModel(),
            elo_model=EloModel(),
            h2h_model=broken_h2h,
            dynamic_factor=DynamicFactorModel(),
        )

        result = ensemble.predict_with_fallback(team_a, team_b)
        win_a, draw, win_b = result
        assert abs(win_a + draw + win_b - 1.0) < 1e-6


# ============================================================================
# TEST: Accuracy report with 0 matches and < 3 matches (Req 5.7, 5.8)
# ============================================================================


class TestAccuracyReportInsufficientData:
    """AccuracyTracker should handle 0 and < 3 match cases."""

    def test_zero_matches_returns_no_data_message(self, data_manager: DataManager) -> None:
        """With zero matches, return 'no data' message."""
        tracker = AccuracyTracker(data_manager)
        result = tracker.calculate_report()

        assert isinstance(result, str)
        assert "無已完賽數據" in result or "無法提供準確度數據" in result

    def test_one_match_returns_insufficient_data(
        self, data_dir: Path, data_manager: DataManager
    ) -> None:
        """With 1 match (< 3), return 'insufficient data' message."""
        # Add 1 match result
        results_data = {
            "matches": [
                {
                    "match_id": "GS-A-1",
                    "date": "2026-06-11",
                    "team_a": "Brazil",
                    "team_b": "Argentina",
                    "score_a": 2,
                    "score_b": 1,
                    "stage": "group",
                    "group": "A",
                    "venue_country": "United States",
                }
            ]
        }
        (data_dir / "match_results.json").write_text(
            json.dumps(results_data), encoding="utf-8"
        )

        # Add 1 matching prediction
        predictions_data = {
            "predictions": [
                {
                    "timestamp": "2026-06-10T14:00:00Z",
                    "match_id": "GS-A-1",
                    "team_a": "Brazil",
                    "team_b": "Argentina",
                    "predicted_score": [1, 0],
                    "win_draw_lose": [0.55, 0.25, 0.20],
                    "confidence_index": 65,
                    "coach_style": "分析師",
                    "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
                }
            ]
        }
        (data_dir / "predictions_log.json").write_text(
            json.dumps(predictions_data), encoding="utf-8"
        )

        tracker = AccuracyTracker(data_manager)
        result = tracker.calculate_report()

        assert isinstance(result, str)
        assert "樣本數不足" in result or "1" in result

    def test_two_matches_returns_insufficient_data(
        self, data_dir: Path, data_manager: DataManager
    ) -> None:
        """With 2 matches (< 3), return 'insufficient data' with count."""
        results_data = {
            "matches": [
                {
                    "match_id": "GS-A-1",
                    "date": "2026-06-11",
                    "team_a": "Brazil",
                    "team_b": "Argentina",
                    "score_a": 2,
                    "score_b": 1,
                    "stage": "group",
                },
                {
                    "match_id": "GS-A-2",
                    "date": "2026-06-12",
                    "team_a": "Germany",
                    "team_b": "France",
                    "score_a": 0,
                    "score_b": 0,
                    "stage": "group",
                },
            ]
        }
        (data_dir / "match_results.json").write_text(
            json.dumps(results_data), encoding="utf-8"
        )

        predictions_data = {
            "predictions": [
                {
                    "timestamp": "2026-06-10T14:00:00Z",
                    "match_id": "GS-A-1",
                    "team_a": "Brazil",
                    "team_b": "Argentina",
                    "predicted_score": [1, 0],
                    "win_draw_lose": [0.55, 0.25, 0.20],
                    "confidence_index": 65,
                    "coach_style": "分析師",
                    "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
                },
                {
                    "timestamp": "2026-06-10T15:00:00Z",
                    "match_id": "GS-A-2",
                    "team_a": "Germany",
                    "team_b": "France",
                    "predicted_score": [1, 1],
                    "win_draw_lose": [0.35, 0.30, 0.35],
                    "confidence_index": 45,
                    "coach_style": "分析師",
                    "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
                },
            ]
        }
        (data_dir / "predictions_log.json").write_text(
            json.dumps(predictions_data), encoding="utf-8"
        )

        tracker = AccuracyTracker(data_manager)
        result = tracker.calculate_report()

        assert isinstance(result, str)
        assert "樣本數不足" in result
        assert "2" in result  # Should mention the current count


# ============================================================================
# TEST: Recalibration systematic bias report (Req 4.8)
# ============================================================================


class TestRecalibrationBiasReport:
    """
    After 5+ matches with < 50% direction accuracy,
    RecalibrationProcess should trigger a systematic bias report.
    """

    @pytest.mark.asyncio
    async def test_bias_report_triggered_on_low_direction_accuracy(
        self, data_dir: Path, data_manager: DataManager
    ) -> None:
        """When direction accuracy < 50% after 5+ matches, bias report is generated."""
        # Create 6 match results where predictions are mostly wrong in direction
        match_results = []
        predictions = []

        for i in range(6):
            match_id = f"GS-A-{i+1}"
            # Actual: team_a always wins (score 2-0)
            match_results.append(
                MatchResult(
                    match_id=match_id,
                    date=f"2026-06-{11+i}",
                    team_a="Brazil",
                    team_b="Argentina",
                    score_a=2,
                    score_b=0,
                    stage="group",
                    group="A",
                    venue_country="United States",
                )
            )
            # Prediction: incorrectly predicts team_b wins (direction wrong)
            # Only first 2 predictions are correct direction to get < 50%
            if i < 2:
                predicted_score = [2, 1]  # correct direction (win_a)
                wdl = [0.55, 0.25, 0.20]
            else:
                predicted_score = [0, 2]  # wrong direction (win_b)
                wdl = [0.20, 0.25, 0.55]

            predictions.append({
                "timestamp": f"2026-06-{10+i}T14:00:00Z",
                "match_id": match_id,
                "team_a": "Brazil",
                "team_b": "Argentina",
                "predicted_score": predicted_score,
                "win_draw_lose": wdl,
                "confidence_index": 60,
                "coach_style": "分析師",
                "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
            })

        # Write match results
        results_data = {
            "matches": [
                {
                    "match_id": r.match_id,
                    "date": r.date,
                    "team_a": r.team_a,
                    "team_b": r.team_b,
                    "score_a": r.score_a,
                    "score_b": r.score_b,
                    "stage": r.stage,
                    "group": r.group,
                    "venue_country": r.venue_country,
                }
                for r in match_results
            ]
        }
        (data_dir / "match_results.json").write_text(
            json.dumps(results_data), encoding="utf-8"
        )

        # Write predictions log
        (data_dir / "predictions_log.json").write_text(
            json.dumps({"predictions": predictions}), encoding="utf-8"
        )

        # Write teams.json with at least the two teams used
        teams_data = {"teams": []}
        for team_name, conf in [("Brazil", "CONMEBOL"), ("Argentina", "CONMEBOL")]:
            team = _make_team(team_name, conf)
            teams_data["teams"].append({
                "name": team.name,
                "name_zh": team.name_zh,
                "aliases": team.aliases,
                "confederation": team.confederation,
                "fifa_ranking": team.fifa_ranking,
                "fifa_points": team.fifa_points,
                "elo_rating": team.elo_rating,
                "group": team.group,
                "recent_goals_avg": team.recent_goals_avg,
                "recent_conceded_avg": team.recent_conceded_avg,
                "recent_win_rate": team.recent_win_rate,
                "recent_draw_rate": team.recent_draw_rate,
                "recent_loss_rate": team.recent_loss_rate,
                "neutral_win_rate": team.neutral_win_rate,
                "best_wc_result": team.best_wc_result,
                "vs_top20_win_rate": team.vs_top20_win_rate,
                "wc_first_match_win_rate": team.wc_first_match_win_rate,
                "penalty_shootout_win_rate": team.penalty_shootout_win_rate,
                "first_half_goal_pct": team.first_half_goal_pct,
                "second_half_goal_pct": team.second_half_goal_pct,
                "clean_sheet_rate": team.clean_sheet_rate,
                "failed_to_score_rate": team.failed_to_score_rate,
                "current_win_streak": team.current_win_streak,
                "current_loss_streak": team.current_loss_streak,
                "last_match_date": team.last_match_date,
                "eliminated_by_2022": team.eliminated_by_2022,
            })
        (data_dir / "teams.json").write_text(
            json.dumps(teams_data), encoding="utf-8"
        )

        # Run recalibration
        ensemble = EnsembleModel(weights=EnsembleWeights())
        process = RecalibrationProcess(data_manager=data_manager, ensemble=ensemble)

        report = await process.update_results(match_results=match_results)

        # Verify: 6 matches evaluated, direction hits = 2 → 33% < 50% → bias report triggered
        assert report.matches_evaluated == 6
        assert report.direction_hits == 2  # Only first 2 are correct
        assert report.systematic_bias is not None
        assert "偏差" in report.systematic_bias or "bias" in report.systematic_bias.lower()

    @pytest.mark.asyncio
    async def test_no_bias_report_when_direction_accuracy_above_50(
        self, data_dir: Path, data_manager: DataManager
    ) -> None:
        """When direction accuracy >= 50% after 5+ matches, no bias report."""
        match_results = []
        predictions = []

        for i in range(6):
            match_id = f"GS-B-{i+1}"
            match_results.append(
                MatchResult(
                    match_id=match_id,
                    date=f"2026-06-{11+i}",
                    team_a="Germany",
                    team_b="France",
                    score_a=1,
                    score_b=0,
                    stage="group",
                    group="B",
                    venue_country="United States",
                )
            )
            # 4 out of 6 predictions correct direction (67% > 50%)
            if i < 4:
                predicted_score = [2, 1]  # correct (win_a)
                wdl = [0.55, 0.25, 0.20]
            else:
                predicted_score = [0, 1]  # wrong (win_b)
                wdl = [0.20, 0.25, 0.55]

            predictions.append({
                "timestamp": f"2026-06-{10+i}T14:00:00Z",
                "match_id": match_id,
                "team_a": "Germany",
                "team_b": "France",
                "predicted_score": predicted_score,
                "win_draw_lose": wdl,
                "confidence_index": 60,
                "coach_style": "分析師",
                "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
            })

        (data_dir / "match_results.json").write_text(
            json.dumps({
                "matches": [
                    {
                        "match_id": r.match_id,
                        "date": r.date,
                        "team_a": r.team_a,
                        "team_b": r.team_b,
                        "score_a": r.score_a,
                        "score_b": r.score_b,
                        "stage": r.stage,
                        "group": r.group,
                        "venue_country": r.venue_country,
                    }
                    for r in match_results
                ]
            }),
            encoding="utf-8",
        )
        (data_dir / "predictions_log.json").write_text(
            json.dumps({"predictions": predictions}), encoding="utf-8"
        )

        # Write teams.json
        teams_data = {"teams": []}
        for team_name, conf in [("Germany", "UEFA"), ("France", "UEFA")]:
            team = _make_team(team_name, conf)
            teams_data["teams"].append({
                "name": team.name,
                "name_zh": team.name_zh,
                "aliases": team.aliases,
                "confederation": team.confederation,
                "fifa_ranking": team.fifa_ranking,
                "fifa_points": team.fifa_points,
                "elo_rating": team.elo_rating,
                "group": team.group,
                "recent_goals_avg": team.recent_goals_avg,
                "recent_conceded_avg": team.recent_conceded_avg,
                "recent_win_rate": team.recent_win_rate,
                "recent_draw_rate": team.recent_draw_rate,
                "recent_loss_rate": team.recent_loss_rate,
                "neutral_win_rate": team.neutral_win_rate,
                "best_wc_result": team.best_wc_result,
                "vs_top20_win_rate": team.vs_top20_win_rate,
                "wc_first_match_win_rate": team.wc_first_match_win_rate,
                "penalty_shootout_win_rate": team.penalty_shootout_win_rate,
                "first_half_goal_pct": team.first_half_goal_pct,
                "second_half_goal_pct": team.second_half_goal_pct,
                "clean_sheet_rate": team.clean_sheet_rate,
                "failed_to_score_rate": team.failed_to_score_rate,
                "current_win_streak": team.current_win_streak,
                "current_loss_streak": team.current_loss_streak,
                "last_match_date": team.last_match_date,
                "eliminated_by_2022": team.eliminated_by_2022,
            })
        (data_dir / "teams.json").write_text(
            json.dumps(teams_data), encoding="utf-8"
        )

        ensemble = EnsembleModel(weights=EnsembleWeights())
        process = RecalibrationProcess(data_manager=data_manager, ensemble=ensemble)
        report = await process.update_results(match_results=match_results)

        assert report.matches_evaluated == 6
        assert report.direction_hits == 4  # 67% > 50%
        assert report.systematic_bias is None

    @pytest.mark.asyncio
    async def test_no_bias_report_when_less_than_5_matches(
        self, data_dir: Path, data_manager: DataManager
    ) -> None:
        """When < 5 matches, no bias report even if direction accuracy is 0%."""
        match_results = []
        predictions = []

        for i in range(4):  # Only 4 matches (below threshold)
            match_id = f"GS-C-{i+1}"
            match_results.append(
                MatchResult(
                    match_id=match_id,
                    date=f"2026-06-{11+i}",
                    team_a="Spain",
                    team_b="Portugal",
                    score_a=3,
                    score_b=0,
                    stage="group",
                    group="C",
                    venue_country="United States",
                )
            )
            # All predictions wrong direction
            predictions.append({
                "timestamp": f"2026-06-{10+i}T14:00:00Z",
                "match_id": match_id,
                "team_a": "Spain",
                "team_b": "Portugal",
                "predicted_score": [0, 2],  # Wrong direction
                "win_draw_lose": [0.20, 0.25, 0.55],
                "confidence_index": 60,
                "coach_style": "分析師",
                "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
            })

        (data_dir / "match_results.json").write_text(
            json.dumps({
                "matches": [
                    {
                        "match_id": r.match_id,
                        "date": r.date,
                        "team_a": r.team_a,
                        "team_b": r.team_b,
                        "score_a": r.score_a,
                        "score_b": r.score_b,
                        "stage": r.stage,
                        "group": r.group,
                        "venue_country": r.venue_country,
                    }
                    for r in match_results
                ]
            }),
            encoding="utf-8",
        )
        (data_dir / "predictions_log.json").write_text(
            json.dumps({"predictions": predictions}), encoding="utf-8"
        )

        # Write teams.json
        teams_data = {"teams": []}
        for team_name, conf in [("Spain", "UEFA"), ("Portugal", "UEFA")]:
            team = _make_team(team_name, conf)
            teams_data["teams"].append({
                "name": team.name,
                "name_zh": team.name_zh,
                "aliases": team.aliases,
                "confederation": team.confederation,
                "fifa_ranking": team.fifa_ranking,
                "fifa_points": team.fifa_points,
                "elo_rating": team.elo_rating,
                "group": team.group,
                "recent_goals_avg": team.recent_goals_avg,
                "recent_conceded_avg": team.recent_conceded_avg,
                "recent_win_rate": team.recent_win_rate,
                "recent_draw_rate": team.recent_draw_rate,
                "recent_loss_rate": team.recent_loss_rate,
                "neutral_win_rate": team.neutral_win_rate,
                "best_wc_result": team.best_wc_result,
                "vs_top20_win_rate": team.vs_top20_win_rate,
                "wc_first_match_win_rate": team.wc_first_match_win_rate,
                "penalty_shootout_win_rate": team.penalty_shootout_win_rate,
                "first_half_goal_pct": team.first_half_goal_pct,
                "second_half_goal_pct": team.second_half_goal_pct,
                "clean_sheet_rate": team.clean_sheet_rate,
                "failed_to_score_rate": team.failed_to_score_rate,
                "current_win_streak": team.current_win_streak,
                "current_loss_streak": team.current_loss_streak,
                "last_match_date": team.last_match_date,
                "eliminated_by_2022": team.eliminated_by_2022,
            })
        (data_dir / "teams.json").write_text(
            json.dumps(teams_data), encoding="utf-8"
        )

        ensemble = EnsembleModel(weights=EnsembleWeights())
        process = RecalibrationProcess(data_manager=data_manager, ensemble=ensemble)
        report = await process.update_results(match_results=match_results)

        assert report.matches_evaluated == 4
        assert report.direction_hits == 0  # All wrong
        # But < 5 matches, so no bias report
        assert report.systematic_bias is None
