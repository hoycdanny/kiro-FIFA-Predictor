"""
Tests for src/tools/predict_match.py - handle_predict_match function.

Verifies:
- Input validation (invalid teams, invalid styles)
- Successful prediction flow
- All three coach styles displayed in output (Requirement 8.4)
- Default analyst style when none specified (Requirement 8.5)
- Prediction logging to data_manager
- Error handling
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.tools.predict_match import handle_predict_match, _format_error
from src.data.data_manager import PredictionError
from src.engine.prediction_engine import MatchPrediction


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_validator():
    """Create a mock InputValidator."""
    validator = MagicMock()
    # Default: valid teams
    validator.validate_team.side_effect = lambda name: name
    return validator


@pytest.fixture
def mock_engine():
    """Create a mock PredictionEngine that returns realistic predictions."""
    engine = MagicMock()

    def make_prediction(team_a, team_b, coach_style=None):
        return MatchPrediction(
            team_a=team_a,
            team_b=team_b,
            win_prob=0.45,
            draw_prob=0.25,
            lose_prob=0.30,
            top_scores=[(1, 0, 0.15), (1, 1, 0.12), (2, 1, 0.10)],
            confidence_index=55,
            over_2_5=0.45,
            under_2_5=0.55,
            expected_goals_a=1.35,
            expected_goals_b=1.10,
            coach_style=coach_style or "分析師",
        )

    engine.predict_match.side_effect = make_prediction

    # Mock coach_style system for narrative generation
    engine.coach_style = MagicMock()
    engine.coach_style.generate_narrative.return_value = "根據統計分析…test narrative"

    # Mock _get_team for _prediction_to_simple
    mock_profile = MagicMock()
    mock_profile.current_win_streak = 0
    mock_profile.current_loss_streak = 0
    mock_profile.last_match_date = None
    mock_profile.eliminated_by_2022 = None
    engine._get_team.return_value = mock_profile
    engine._get_days_rest.return_value = 7

    return engine


@pytest.fixture
def mock_formatter():
    """Create a mock OutputFormatter."""
    formatter = MagicMock()
    formatter.format_match_prediction.return_value = "## ⚽ Formatted Prediction Output"
    return formatter


@pytest.fixture
def mock_data_manager():
    """Create a mock DataManager."""
    dm = MagicMock()
    dm.append_prediction_log.return_value = None
    return dm


# ============================================================================
# TEST CLASSES
# ============================================================================


class TestHandlePredictMatchValidation:
    """Test input validation scenarios."""

    @pytest.mark.asyncio
    async def test_returns_error_when_server_not_initialized(self):
        """Should return init error when dependencies are None."""
        result = await handle_predict_match(
            team_a="Brazil",
            team_b="Argentina",
            coach_style=None,
            engine=None,
            validator=None,
            formatter=None,
            data_manager=None,
        )
        assert "Error" in result
        assert "not initialized" in result

    @pytest.mark.asyncio
    async def test_returns_error_for_invalid_team_a(self, mock_engine, mock_formatter, mock_data_manager):
        """Should return error with suggestions for invalid team A."""
        validator = MagicMock()
        validator.validate_team.return_value = PredictionError(
            error_code="INVALID_TEAM",
            message="找不到球隊「Foobar」，不在 48 支參賽隊伍中。",
            suggestions=["France", "Finland"],
        )

        result = await handle_predict_match(
            team_a="Foobar",
            team_b="Brazil",
            coach_style=None,
            engine=mock_engine,
            validator=validator,
            formatter=mock_formatter,
            data_manager=mock_data_manager,
        )

        assert "❌" in result
        assert "Foobar" in result
        assert "France" in result

    @pytest.mark.asyncio
    async def test_returns_error_for_invalid_team_b(self, mock_engine, mock_formatter, mock_data_manager):
        """Should return error with suggestions for invalid team B."""
        validator = MagicMock()
        # First call (team_a) succeeds, second call (team_b) fails
        validator.validate_team.side_effect = [
            "Brazil",
            PredictionError(
                error_code="INVALID_TEAM",
                message="找不到球隊「XYZ」，不在 48 支參賽隊伍中。",
                suggestions=["Mexico"],
            ),
        ]

        result = await handle_predict_match(
            team_a="Brazil",
            team_b="XYZ",
            coach_style=None,
            engine=mock_engine,
            validator=validator,
            formatter=mock_formatter,
            data_manager=mock_data_manager,
        )

        assert "❌" in result
        assert "XYZ" in result

    @pytest.mark.asyncio
    async def test_returns_error_for_invalid_coach_style(
        self, mock_engine, mock_formatter, mock_data_manager
    ):
        """Should return error with valid style options for invalid coach style."""
        validator = MagicMock()
        validator.validate_team.side_effect = lambda name: name
        validator.validate_coach_style.return_value = PredictionError(
            error_code="INVALID_STYLE",
            message="教練風格「badstyle」無效。",
            suggestions=["分析師 (analyst)", "反向思考者 (contrarian)", "戰術家 (tactician)"],
        )

        result = await handle_predict_match(
            team_a="Brazil",
            team_b="Argentina",
            coach_style="badstyle",
            engine=mock_engine,
            validator=validator,
            formatter=mock_formatter,
            data_manager=mock_data_manager,
        )

        assert "❌" in result
        assert "badstyle" in result
        assert "分析師" in result


class TestHandlePredictMatchSuccess:
    """Test successful prediction flow."""

    @pytest.mark.asyncio
    async def test_successful_prediction_default_style(
        self, mock_validator, mock_engine, mock_formatter, mock_data_manager
    ):
        """Should return formatted prediction with default analyst style."""
        from src.utils.validator import CoachStyleType

        result = await handle_predict_match(
            team_a="Brazil",
            team_b="Argentina",
            coach_style=None,
            engine=mock_engine,
            validator=mock_validator,
            formatter=mock_formatter,
            data_manager=mock_data_manager,
        )

        # Primary prediction should use analyst style
        assert mock_engine.predict_match.called
        first_call = mock_engine.predict_match.call_args_list[0]
        assert first_call.kwargs["coach_style"] == "分析師"

        # Output should contain the formatter's result
        assert "Formatted Prediction Output" in result

    @pytest.mark.asyncio
    async def test_successful_prediction_specified_style(
        self, mock_validator, mock_engine, mock_formatter, mock_data_manager
    ):
        """Should use validated coach style for primary prediction."""
        from src.utils.validator import CoachStyleType

        mock_validator.validate_coach_style.return_value = CoachStyleType.CONTRARIAN

        result = await handle_predict_match(
            team_a="Brazil",
            team_b="Argentina",
            coach_style="aggressive",
            engine=mock_engine,
            validator=mock_validator,
            formatter=mock_formatter,
            data_manager=mock_data_manager,
        )

        # Primary prediction should use contrarian style
        first_call = mock_engine.predict_match.call_args_list[0]
        assert first_call.kwargs["coach_style"] == "反向思考者"

    @pytest.mark.asyncio
    async def test_output_includes_all_three_coach_styles(
        self, mock_validator, mock_engine, mock_formatter, mock_data_manager
    ):
        """Requirement 8.4: Output should display all three coach style perspectives."""
        result = await handle_predict_match(
            team_a="Brazil",
            team_b="Argentina",
            coach_style=None,
            engine=mock_engine,
            validator=mock_validator,
            formatter=mock_formatter,
            data_manager=mock_data_manager,
        )

        # Should contain all three style labels
        assert "分析師 (Analyst)" in result
        assert "反向思考者 (Contrarian)" in result
        assert "戰術家 (Tactician)" in result

        # Should contain the Coach Style Perspectives header
        assert "Coach Style Perspectives" in result

    @pytest.mark.asyncio
    async def test_primary_style_is_marked_in_output(
        self, mock_validator, mock_engine, mock_formatter, mock_data_manager
    ):
        """Primary style should be marked with indicator in output."""
        result = await handle_predict_match(
            team_a="Brazil",
            team_b="Argentina",
            coach_style=None,
            engine=mock_engine,
            validator=mock_validator,
            formatter=mock_formatter,
            data_manager=mock_data_manager,
        )

        # Default style (analyst) should be marked
        assert "⬅️" in result

    @pytest.mark.asyncio
    async def test_prediction_is_logged(
        self, mock_validator, mock_engine, mock_formatter, mock_data_manager
    ):
        """Prediction should be logged via data_manager.append_prediction_log."""
        await handle_predict_match(
            team_a="Brazil",
            team_b="Argentina",
            coach_style=None,
            engine=mock_engine,
            validator=mock_validator,
            formatter=mock_formatter,
            data_manager=mock_data_manager,
        )

        mock_data_manager.append_prediction_log.assert_called_once()
        log_entry = mock_data_manager.append_prediction_log.call_args[0][0]
        assert log_entry.team_a == "Brazil"
        assert log_entry.team_b == "Argentina"
        assert log_entry.coach_style == "分析師"

    @pytest.mark.asyncio
    async def test_logging_failure_does_not_break_response(
        self, mock_validator, mock_engine, mock_formatter, mock_data_manager
    ):
        """Logging failure should not prevent prediction from returning."""
        mock_data_manager.append_prediction_log.side_effect = Exception("Disk full")

        result = await handle_predict_match(
            team_a="Brazil",
            team_b="Argentina",
            coach_style=None,
            engine=mock_engine,
            validator=mock_validator,
            formatter=mock_formatter,
            data_manager=mock_data_manager,
        )

        # Should still return the formatted prediction
        assert "Formatted Prediction Output" in result


class TestHandlePredictMatchEngineErrors:
    """Test engine error handling."""

    @pytest.mark.asyncio
    async def test_engine_value_error_returns_error_message(
        self, mock_validator, mock_formatter, mock_data_manager
    ):
        """Should return error message if engine raises ValueError."""
        engine = MagicMock()
        engine.predict_match.side_effect = ValueError("Team 'Foo' not found in database.")

        result = await handle_predict_match(
            team_a="Brazil",
            team_b="Argentina",
            coach_style=None,
            engine=engine,
            validator=mock_validator,
            formatter=mock_formatter,
            data_manager=mock_data_manager,
        )

        assert "Error:" in result
        assert "not found in database" in result


class TestFormatError:
    """Test the _format_error helper function."""

    def test_format_error_with_suggestions(self):
        """Should format error with bullet-pointed suggestions."""
        error = PredictionError(
            error_code="INVALID_TEAM",
            message="找不到球隊「Fronce」",
            suggestions=["France", "Finland"],
        )
        result = _format_error(error)

        assert "❌" in result
        assert "Fronce" in result
        assert "France" in result
        assert "Finland" in result
        assert "•" in result

    def test_format_error_without_suggestions(self):
        """Should format error without suggestions section."""
        error = PredictionError(
            error_code="INVALID_TEAM",
            message="球隊名稱不得為空。",
            suggestions=[],
        )
        result = _format_error(error)

        assert "❌" in result
        assert "球隊名稱不得為空" in result
        assert "•" not in result
