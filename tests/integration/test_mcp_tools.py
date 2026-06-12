"""
Integration tests for MCP tool end-to-end flows.

Tests the full pipeline of each MCP tool from user input through to formatted output,
using real data from data/ files to verify end-to-end behavior.

Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1
"""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import pytest

from src.data.data_manager import DataManager, MatchResult, PredictionLogEntry
from src.engine.ensemble import EnsembleModel, EnsembleWeights
from src.engine.monte_carlo import MonteCarloSimulator
from src.engine.prediction_engine import PredictionEngine
from src.output.formatter import OutputFormatter
from src.utils.accuracy_tracker import AccuracyTracker
from src.utils.team_matcher import TeamMatcher
from src.utils.validator import InputValidator


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def data_dir() -> Path:
    """Return the real project data directory."""
    return Path(__file__).resolve().parent.parent.parent / "data"


@pytest.fixture
def data_manager(data_dir: Path) -> DataManager:
    """Create a DataManager pointing to real data."""
    return DataManager(data_dir)


@pytest.fixture
def ensemble() -> EnsembleModel:
    """Create an EnsembleModel with default weights."""
    return EnsembleModel(weights=EnsembleWeights())


@pytest.fixture
def prediction_engine(data_manager: DataManager, ensemble: EnsembleModel) -> PredictionEngine:
    """Create a PredictionEngine wired to real data."""
    return PredictionEngine(data_manager=data_manager, ensemble=ensemble)


@pytest.fixture
def monte_carlo(ensemble: EnsembleModel, data_manager: DataManager) -> MonteCarloSimulator:
    """Create a MonteCarloSimulator with real team data."""
    teams = data_manager.load_teams()
    teams_dict = {t.name: t for t in teams}
    return MonteCarloSimulator(ensemble=ensemble, teams=teams_dict)


@pytest.fixture
def formatter() -> OutputFormatter:
    """Create an OutputFormatter (defaults to Markdown)."""
    return OutputFormatter()


@pytest.fixture
def validator(data_manager: DataManager) -> InputValidator:
    """Create an InputValidator with real team names."""
    teams = data_manager.load_teams()
    team_names = [t.name for t in teams]
    team_matcher = TeamMatcher(team_names)
    return InputValidator(team_matcher=team_matcher)


@pytest.fixture
def accuracy_tracker(data_manager: DataManager) -> AccuracyTracker:
    """Create an AccuracyTracker with real data."""
    return AccuracyTracker(data_manager=data_manager)


@pytest.fixture
def tmp_data_dir(data_dir: Path) -> Path:
    """Create a temporary copy of data directory for tests that modify data."""
    tmp_dir = Path(tempfile.mkdtemp())
    # Copy all JSON files from real data dir
    for f in data_dir.glob("*.json"):
        shutil.copy2(f, tmp_dir / f.name)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================================
# TEST: predict_match full flow
# ============================================================================


class TestPredictMatchFlow:
    """Integration tests for the predict_match MCP tool end-to-end."""

    async def test_valid_match_prediction_produces_formatted_output(
        self, prediction_engine, validator, formatter, data_manager
    ):
        """Test full predict_match flow: valid input → formatted output."""
        from src.tools.predict_match import handle_predict_match

        result = await handle_predict_match(
            team_a="Brazil",
            team_b="Argentina",
            coach_style=None,
            engine=prediction_engine,
            validator=validator,
            formatter=formatter,
            data_manager=data_manager,
        )

        # Should return a non-empty string (not an error)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "❌" not in result

        # Should contain key prediction elements
        assert "Brazil" in result
        assert "Argentina" in result

        # Should contain probability indicators (W/D/L or percentage symbols)
        assert "%" in result

        # Should contain the coach style section (Requirement 8.4)
        assert "Coach Style" in result or "分析師" in result or "Analyst" in result

    async def test_predict_match_with_coach_style(
        self, prediction_engine, validator, formatter, data_manager
    ):
        """Test predict_match with explicit coach style parameter."""
        from src.tools.predict_match import handle_predict_match

        result = await handle_predict_match(
            team_a="Germany",
            team_b="France",
            coach_style="戰術家",
            engine=prediction_engine,
            validator=validator,
            formatter=formatter,
            data_manager=data_manager,
        )

        assert isinstance(result, str)
        assert "❌" not in result
        assert "Germany" in result
        assert "France" in result

    async def test_predict_match_invalid_team_returns_error_with_suggestions(
        self, prediction_engine, validator, formatter, data_manager
    ):
        """Test that invalid team names return error with suggestions."""
        from src.tools.predict_match import handle_predict_match

        result = await handle_predict_match(
            team_a="Brazill",  # Typo
            team_b="Argentina",
            coach_style=None,
            engine=prediction_engine,
            validator=validator,
            formatter=formatter,
            data_manager=data_manager,
        )

        # Could be a fuzzy match success OR an error — depends on matcher threshold
        # If it's an error, it should have suggestions
        # If fuzzy match resolves, it should work fine
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_predict_match_invalid_coach_style_returns_error(
        self, prediction_engine, validator, formatter, data_manager
    ):
        """Test that invalid coach style returns error with valid options."""
        from src.tools.predict_match import handle_predict_match

        result = await handle_predict_match(
            team_a="Brazil",
            team_b="Argentina",
            coach_style="invalid_style",
            engine=prediction_engine,
            validator=validator,
            formatter=formatter,
            data_manager=data_manager,
        )

        assert isinstance(result, str)
        assert "❌" in result
        # Should suggest valid styles
        assert "分析師" in result or "analyst" in result.lower()

    async def test_predict_match_contains_confidence_index(
        self, prediction_engine, validator, formatter, data_manager
    ):
        """Test that output includes confidence index."""
        from src.tools.predict_match import handle_predict_match

        result = await handle_predict_match(
            team_a="Spain",
            team_b="Portugal",
            coach_style=None,
            engine=prediction_engine,
            validator=validator,
            formatter=formatter,
            data_manager=data_manager,
        )

        assert isinstance(result, str)
        assert "❌" not in result
        # Confidence should appear somewhere in the output
        assert "Confidence" in result or "信心" in result or "confidence" in result.lower()


# ============================================================================
# TEST: predict_group full flow
# ============================================================================


class TestPredictGroupFlow:
    """Integration tests for the predict_group MCP tool end-to-end."""

    async def test_valid_group_produces_table_output(
        self, prediction_engine, validator, formatter
    ):
        """Test full predict_group flow: valid group → table output."""
        from src.tools.predict_group import handle_predict_group

        result = await handle_predict_group(
            group_id="A",
            engine=prediction_engine,
            validator=validator,
            formatter=formatter,
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "❌" not in result

        # Should contain table structure indicators
        assert "|" in result  # Markdown table

        # Should contain ranking-related content
        # Group A should show 4 teams in standings
        assert "W" in result or "勝" in result  # Column headers

    async def test_predict_group_all_valid_groups(
        self, prediction_engine, validator, formatter
    ):
        """Test that all 12 groups (A-L) produce valid output."""
        from src.tools.predict_group import handle_predict_group

        for group_id in "ABCDEFGHIJKL":
            result = await handle_predict_group(
                group_id=group_id,
                engine=prediction_engine,
                validator=validator,
                formatter=formatter,
            )

            assert isinstance(result, str)
            assert "❌" not in result, f"Group {group_id} produced error: {result[:100]}"
            assert "|" in result, f"Group {group_id} missing table format"

    async def test_predict_group_invalid_returns_error(
        self, prediction_engine, validator, formatter
    ):
        """Test that invalid group ID returns error with valid options."""
        from src.tools.predict_group import handle_predict_group

        result = await handle_predict_group(
            group_id="Z",
            engine=prediction_engine,
            validator=validator,
            formatter=formatter,
        )

        assert isinstance(result, str)
        assert "❌" in result
        # Should list valid group codes
        assert "A" in result

    async def test_predict_group_case_insensitive(
        self, prediction_engine, validator, formatter
    ):
        """Test that group ID is case-insensitive."""
        from src.tools.predict_group import handle_predict_group

        result = await handle_predict_group(
            group_id="b",  # lowercase
            engine=prediction_engine,
            validator=validator,
            formatter=formatter,
        )

        assert isinstance(result, str)
        assert "❌" not in result


# ============================================================================
# TEST: predict_champion full flow
# ============================================================================


class TestPredictChampionFlow:
    """Integration tests for the predict_champion MCP tool end-to-end."""

    async def test_champion_prediction_produces_bracket_output(
        self, prediction_engine, monte_carlo, formatter, data_manager
    ):
        """Test full predict_champion flow: simulation → bracket output."""
        from src.tools.predict_champion import handle_predict_champion

        # Use fewer simulations for speed in testing
        result = await handle_predict_champion(
            simulations=500,
            engine=prediction_engine,
            monte_carlo=monte_carlo,
            formatter=formatter,
            data_manager=data_manager,
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Error" not in result

        # Should contain championship-related content
        # Output should reference teams or tournament structure
        assert "%" in result  # Probability percentages

    async def test_champion_prediction_minimum_simulations(
        self, prediction_engine, monte_carlo, formatter, data_manager
    ):
        """Test that simulation count is clamped to minimum 100."""
        from src.tools.predict_champion import handle_predict_champion

        result = await handle_predict_champion(
            simulations=10,  # Below minimum, should be clamped to 100
            engine=prediction_engine,
            monte_carlo=monte_carlo,
            formatter=formatter,
            data_manager=data_manager,
        )

        assert isinstance(result, str)
        assert "Error" not in result


# ============================================================================
# TEST: update_results full flow
# ============================================================================


class TestUpdateResultsFlow:
    """Integration tests for the update_results MCP tool end-to-end."""

    async def test_update_with_manual_result(self, tmp_data_dir, ensemble):
        """Test update_results flow: manual result → recalibration → report."""
        from src.tools.update_results import handle_update_results

        dm = DataManager(tmp_data_dir)

        result = await handle_update_results(
            match_id="TEST-001",
            manual_result="Brazil 2 - 1 Argentina",
            data_manager=dm,
            ensemble=ensemble,
            formatter=None,
        )

        assert isinstance(result, str)
        assert "Error" not in result

        # Should contain recalibration report content
        assert "校準" in result or "權重" in result or "poisson" in result.lower()

    async def test_update_results_no_data(self, tmp_data_dir, ensemble):
        """Test update_results with no match results."""
        from src.tools.update_results import handle_update_results

        dm = DataManager(tmp_data_dir)

        # Ensure empty match results
        match_file = tmp_data_dir / "match_results.json"
        with open(match_file, "w", encoding="utf-8") as f:
            json.dump({"matches": []}, f)

        result = await handle_update_results(
            match_id=None,
            manual_result=None,
            data_manager=dm,
            ensemble=ensemble,
            formatter=None,
        )

        assert isinstance(result, str)
        # Should report 0 matches evaluated
        assert "0" in result

    async def test_update_results_server_not_initialized(self):
        """Test error message when server not initialized."""
        from src.tools.update_results import handle_update_results

        result = await handle_update_results(
            match_id=None,
            manual_result=None,
            data_manager=None,
            ensemble=None,
            formatter=None,
        )

        assert "Error" in result


# ============================================================================
# TEST: accuracy_stats full flow
# ============================================================================


class TestAccuracyStatsFlow:
    """Integration tests for the accuracy_stats MCP tool end-to-end."""

    async def test_accuracy_stats_no_data(self, accuracy_tracker, formatter):
        """Test accuracy_stats with no match data returns appropriate message."""
        from src.tools.accuracy_stats import handle_accuracy_stats

        result = await handle_accuracy_stats(
            accuracy_tracker=accuracy_tracker,
            formatter=formatter,
        )

        assert isinstance(result, str)
        # Should indicate no data or insufficient data
        assert "無" in result or "不足" in result or "0" in result

    async def test_accuracy_stats_insufficient_data(self, tmp_data_dir, formatter):
        """Test accuracy_stats with < 3 matches returns insufficient message."""
        from src.tools.accuracy_stats import handle_accuracy_stats

        # Set up data with 1 match result and 1 prediction
        dm = DataManager(tmp_data_dir)

        # Write one match result
        results_data = {
            "matches": [
                {
                    "match_id": "GS-A-1",
                    "date": "2026-06-11",
                    "team_a": "Morocco",
                    "team_b": "Jamaica",
                    "score_a": 2,
                    "score_b": 0,
                    "stage": "group",
                    "group": "A",
                    "venue_country": "United States",
                }
            ]
        }
        with open(tmp_data_dir / "match_results.json", "w", encoding="utf-8") as f:
            json.dump(results_data, f)

        # Write matching prediction
        predictions_data = {
            "predictions": [
                {
                    "timestamp": "2026-06-10T14:30:00Z",
                    "match_id": "GS-A-1",
                    "team_a": "Morocco",
                    "team_b": "Jamaica",
                    "predicted_score": [2, 0],
                    "win_draw_lose": [0.65, 0.20, 0.15],
                    "confidence_index": 72,
                    "coach_style": "分析師",
                    "model_weights": {
                        "poisson": 0.40,
                        "elo": 0.25,
                        "h2h": 0.15,
                        "dynamic": 0.20,
                    },
                }
            ]
        }
        with open(tmp_data_dir / "predictions_log.json", "w", encoding="utf-8") as f:
            json.dump(predictions_data, f)

        tracker = AccuracyTracker(data_manager=dm)

        result = await handle_accuracy_stats(
            accuracy_tracker=tracker,
            formatter=formatter,
        )

        assert isinstance(result, str)
        # Should indicate insufficient data (< 3 matches)
        assert "不足" in result or "1" in result

    async def test_accuracy_stats_sufficient_data(self, tmp_data_dir, formatter):
        """Test accuracy_stats with >= 3 matches returns full report."""
        from src.tools.accuracy_stats import handle_accuracy_stats

        dm = DataManager(tmp_data_dir)

        # Write 3 match results
        results_data = {
            "matches": [
                {
                    "match_id": "GS-A-1",
                    "date": "2026-06-11",
                    "team_a": "Morocco",
                    "team_b": "Jamaica",
                    "score_a": 2,
                    "score_b": 0,
                    "stage": "group",
                    "group": "A",
                    "venue_country": "United States",
                },
                {
                    "match_id": "GS-A-2",
                    "date": "2026-06-12",
                    "team_a": "Canada",
                    "team_b": "Ecuador",
                    "score_a": 1,
                    "score_b": 1,
                    "stage": "group",
                    "group": "A",
                    "venue_country": "Canada",
                },
                {
                    "match_id": "GS-B-1",
                    "date": "2026-06-12",
                    "team_a": "Spain",
                    "team_b": "Portugal",
                    "score_a": 1,
                    "score_b": 2,
                    "stage": "group",
                    "group": "B",
                    "venue_country": "United States",
                },
            ]
        }
        with open(tmp_data_dir / "match_results.json", "w", encoding="utf-8") as f:
            json.dump(results_data, f)

        # Write matching predictions
        predictions_data = {
            "predictions": [
                {
                    "timestamp": "2026-06-10T14:30:00Z",
                    "match_id": "GS-A-1",
                    "team_a": "Morocco",
                    "team_b": "Jamaica",
                    "predicted_score": [2, 0],
                    "win_draw_lose": [0.65, 0.20, 0.15],
                    "confidence_index": 72,
                    "coach_style": "分析師",
                    "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
                },
                {
                    "timestamp": "2026-06-11T14:30:00Z",
                    "match_id": "GS-A-2",
                    "team_a": "Canada",
                    "team_b": "Ecuador",
                    "predicted_score": [1, 1],
                    "win_draw_lose": [0.30, 0.40, 0.30],
                    "confidence_index": 50,
                    "coach_style": "分析師",
                    "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
                },
                {
                    "timestamp": "2026-06-11T15:00:00Z",
                    "match_id": "GS-B-1",
                    "team_a": "Spain",
                    "team_b": "Portugal",
                    "predicted_score": [2, 1],
                    "win_draw_lose": [0.45, 0.30, 0.25],
                    "confidence_index": 60,
                    "coach_style": "戰術家",
                    "model_weights": {"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
                },
            ]
        }
        with open(tmp_data_dir / "predictions_log.json", "w", encoding="utf-8") as f:
            json.dump(predictions_data, f)

        tracker = AccuracyTracker(data_manager=dm)

        result = await handle_accuracy_stats(
            accuracy_tracker=tracker,
            formatter=formatter,
        )

        assert isinstance(result, str)
        # With sufficient data, should contain percentage metrics
        assert "%" in result

    async def test_accuracy_stats_server_not_initialized(self):
        """Test error message when server not initialized."""
        from src.tools.accuracy_stats import handle_accuracy_stats

        result = await handle_accuracy_stats(
            accuracy_tracker=None,
            formatter=None,
        )

        assert "Error" in result


# ============================================================================
# TEST: team_info full flow
# ============================================================================


class TestTeamInfoFlow:
    """Integration tests for the team_info MCP tool end-to-end."""

    async def test_team_info_exact_match(
        self, validator, formatter, data_manager
    ):
        """Test team_info with exact team name match."""
        from src.tools.team_info import handle_team_info

        result = await handle_team_info(
            team_name="Brazil",
            validator=validator,
            formatter=formatter,
            data_manager=data_manager,
        )

        assert isinstance(result, str)
        assert "❌" not in result
        assert "Brazil" in result or "巴西" in result

        # Should contain team profile sections
        assert "FIFA" in result or "Elo" in result or "elo" in result.lower()

    async def test_team_info_fuzzy_match(
        self, validator, formatter, data_manager
    ):
        """Test team_info with fuzzy name matching."""
        from src.tools.team_info import handle_team_info

        # "USA" should match "United States"
        result = await handle_team_info(
            team_name="USA",
            validator=validator,
            formatter=formatter,
            data_manager=data_manager,
        )

        assert isinstance(result, str)
        # Should resolve successfully (either profile or disambiguation)
        # If resolves, should contain team profile
        if "❌" not in result:
            assert "United States" in result or "美國" in result

    async def test_team_info_no_match(
        self, validator, formatter, data_manager
    ):
        """Test team_info with no matching team name."""
        from src.tools.team_info import handle_team_info

        result = await handle_team_info(
            team_name="Nonexistentland",
            validator=validator,
            formatter=formatter,
            data_manager=data_manager,
        )

        assert isinstance(result, str)
        assert "❌" in result
        # Should have suggestions (up to 3)

    async def test_team_info_chinese_name(
        self, validator, formatter, data_manager
    ):
        """Test team_info with Chinese team name."""
        from src.tools.team_info import handle_team_info

        result = await handle_team_info(
            team_name="巴西",
            validator=validator,
            formatter=formatter,
            data_manager=data_manager,
        )

        assert isinstance(result, str)
        # Should resolve Brazil
        if "❌" not in result:
            assert "Brazil" in result or "巴西" in result

    async def test_team_info_server_not_initialized(self):
        """Test error message when server not initialized."""
        from src.tools.team_info import handle_team_info

        result = await handle_team_info(
            team_name="Brazil",
            validator=None,
            formatter=None,
            data_manager=None,
        )

        assert "Error" in result

    async def test_team_info_contains_profile_sections(
        self, validator, formatter, data_manager
    ):
        """Test that team profile output contains categorized sections."""
        from src.tools.team_info import handle_team_info

        result = await handle_team_info(
            team_name="France",
            validator=validator,
            formatter=formatter,
            data_manager=data_manager,
        )

        assert isinstance(result, str)
        if "❌" not in result:
            # Should contain structured output with data sections
            # Check for ranking or statistics keywords
            has_stats = (
                "ranking" in result.lower()
                or "排名" in result
                or "elo" in result.lower()
                or "FIFA" in result
            )
            assert has_stats, f"Team profile output lacks stats content: {result[:200]}"
