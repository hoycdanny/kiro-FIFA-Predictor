"""Unit tests for output formatter and markdown renderer."""

import pytest

from src.data.data_manager import TeamProfile
from src.engine.monte_carlo import ChampionPrediction
from src.engine.prediction_engine import (
    GroupPrediction,
    GroupStanding,
    MatchPrediction,
)
from src.output.formatter import MODEL_VERSION, OutputFormatter
from src.output.markdown_renderer import FLAG_EMOJI, MarkdownRenderer


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def formatter() -> OutputFormatter:
    """Create OutputFormatter with default MarkdownRenderer."""
    return OutputFormatter()


@pytest.fixture
def sample_match_prediction() -> MatchPrediction:
    """Create a sample match prediction for testing."""
    return MatchPrediction(
        team_a="Brazil",
        team_b="Argentina",
        win_prob=0.45,
        draw_prob=0.25,
        lose_prob=0.30,
        top_scores=[(1, 0, 0.12), (2, 1, 0.10), (1, 1, 0.09)],
        confidence_index=72,
        over_2_5=0.55,
        under_2_5=0.45,
        expected_goals_a=1.52,
        expected_goals_b=1.18,
        coach_style="分析師",
    )


@pytest.fixture
def sample_group_prediction() -> GroupPrediction:
    """Create a sample group prediction for testing."""
    standings = [
        GroupStanding(
            team="Brazil",
            played=3,
            wins=2,
            draws=1,
            losses=0,
            goals_for=5,
            goals_against=2,
            goal_difference=3,
            points=7,
            qualification_status="確定晉級",
        ),
        GroupStanding(
            team="Germany",
            played=3,
            wins=2,
            draws=0,
            losses=1,
            goals_for=4,
            goals_against=3,
            goal_difference=1,
            points=6,
            qualification_status="確定晉級",
        ),
        GroupStanding(
            team="Japan",
            played=3,
            wins=1,
            draws=0,
            losses=2,
            goals_for=2,
            goals_against=4,
            goal_difference=-2,
            points=3,
            qualification_status="可能晉級 (55%)",
        ),
        GroupStanding(
            team="Morocco",
            played=3,
            wins=0,
            draws=1,
            losses=2,
            goals_for=1,
            goals_against=3,
            goal_difference=-2,
            points=1,
            qualification_status="",
        ),
    ]
    return GroupPrediction(
        group_id="A",
        standings=standings,
        match_predictions=[],
    )


@pytest.fixture
def sample_champion_prediction() -> ChampionPrediction:
    """Create a sample champion prediction for testing."""
    return ChampionPrediction(
        predicted_champion="Brazil",
        champion_probability=0.18,
        top_5=[
            ("Brazil", 0.18),
            ("France", 0.15),
            ("Argentina", 0.12),
            ("England", 0.09),
            ("Spain", 0.08),
        ],
        round_probabilities={
            "Brazil": {
                "round_of_32": 0.95,
                "round_of_16": 0.82,
                "quarter_finals": 0.60,
                "semi_finals": 0.40,
                "final": 0.25,
                "champion": 0.18,
            },
            "France": {
                "round_of_32": 0.93,
                "round_of_16": 0.78,
                "quarter_finals": 0.55,
                "semi_finals": 0.35,
                "final": 0.22,
                "champion": 0.15,
            },
        },
        confidence_index=65,
    )


@pytest.fixture
def sample_team_profile() -> TeamProfile:
    """Create a sample team profile for testing."""
    return TeamProfile(
        name="Brazil",
        name_zh="巴西",
        aliases=["BRA", "Brasil"],
        confederation="CONMEBOL",
        fifa_ranking=1,
        fifa_points=1840.5,
        elo_rating=2150,
        group="A",
        recent_goals_avg=2.10,
        recent_conceded_avg=0.80,
        recent_win_rate=70.0,
        recent_draw_rate=20.0,
        recent_loss_rate=10.0,
        neutral_win_rate=65.0,
        best_wc_result="Champion (5x)",
        vs_top20_win_rate=55.0,
        wc_first_match_win_rate=72.0,
        penalty_shootout_win_rate=60.0,
        first_half_goal_pct=45.0,
        second_half_goal_pct=55.0,
        clean_sheet_rate=40.0,
        failed_to_score_rate=10.0,
        current_win_streak=3,
        current_loss_streak=0,
    )


# ============================================================================
# MATCH PREDICTION TESTS
# ============================================================================


class TestMatchPredictionOutput:
    """Tests for match prediction formatting."""

    def test_contains_team_names(
        self, formatter: OutputFormatter, sample_match_prediction: MatchPrediction
    ):
        output = formatter.format_match_prediction(
            sample_match_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "Brazil" in output
        assert "Argentina" in output

    def test_contains_flag_emojis(
        self, formatter: OutputFormatter, sample_match_prediction: MatchPrediction
    ):
        output = formatter.format_match_prediction(
            sample_match_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "🇧🇷" in output
        assert "🇦🇷" in output

    def test_contains_probabilities(
        self, formatter: OutputFormatter, sample_match_prediction: MatchPrediction
    ):
        output = formatter.format_match_prediction(
            sample_match_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "45.0%" in output
        assert "25.0%" in output
        assert "30.0%" in output

    def test_contains_top_scores(
        self, formatter: OutputFormatter, sample_match_prediction: MatchPrediction
    ):
        output = formatter.format_match_prediction(
            sample_match_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "1-0" in output
        assert "2-1" in output
        assert "1-1" in output

    def test_contains_confidence(
        self, formatter: OutputFormatter, sample_match_prediction: MatchPrediction
    ):
        output = formatter.format_match_prediction(
            sample_match_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "72%" in output

    def test_contains_xg(
        self, formatter: OutputFormatter, sample_match_prediction: MatchPrediction
    ):
        output = formatter.format_match_prediction(
            sample_match_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "1.52" in output
        assert "1.18" in output

    def test_contains_over_under(
        self, formatter: OutputFormatter, sample_match_prediction: MatchPrediction
    ):
        output = formatter.format_match_prediction(
            sample_match_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "55.0%" in output
        assert "45.0%" in output
        assert "Over 2.5" in output
        assert "Under 2.5" in output


# ============================================================================
# GROUP STANDINGS TESTS
# ============================================================================


class TestGroupStandingsOutput:
    """Tests for group standings formatting."""

    def test_contains_markdown_table_headers(
        self, formatter: OutputFormatter, sample_group_prediction: GroupPrediction
    ):
        output = formatter.format_group_standings(
            sample_group_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "| Rank |" in output
        assert "| Team |" in output or "Team" in output
        assert "| P |" in output or "P" in output
        assert "| W |" in output or "W" in output
        assert "| D |" in output or "D" in output
        assert "| L |" in output or "L" in output
        assert "GF" in output
        assert "GA" in output
        assert "GD" in output

    def test_contains_team_names(
        self, formatter: OutputFormatter, sample_group_prediction: GroupPrediction
    ):
        output = formatter.format_group_standings(
            sample_group_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "Brazil" in output
        assert "Germany" in output
        assert "Japan" in output
        assert "Morocco" in output

    def test_contains_group_identifier(
        self, formatter: OutputFormatter, sample_group_prediction: GroupPrediction
    ):
        output = formatter.format_group_standings(
            sample_group_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "Group A" in output

    def test_contains_table_separator(
        self, formatter: OutputFormatter, sample_group_prediction: GroupPrediction
    ):
        output = formatter.format_group_standings(
            sample_group_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        # Markdown table separator row
        assert "|---" in output


# ============================================================================
# CHAMPION PREDICTION TESTS
# ============================================================================


class TestChampionPredictionOutput:
    """Tests for champion prediction formatting."""

    def test_contains_champion_name(
        self, formatter: OutputFormatter, sample_champion_prediction: ChampionPrediction
    ):
        output = formatter.format_champion_prediction(
            sample_champion_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "Brazil" in output

    def test_contains_top_5_teams(
        self, formatter: OutputFormatter, sample_champion_prediction: ChampionPrediction
    ):
        output = formatter.format_champion_prediction(
            sample_champion_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "France" in output
        assert "Argentina" in output
        assert "England" in output
        assert "Spain" in output


# ============================================================================
# TEAM PROFILE TESTS
# ============================================================================


class TestTeamProfileOutput:
    """Tests for team profile formatting."""

    def test_contains_section_headers(
        self, formatter: OutputFormatter, sample_team_profile: TeamProfile
    ):
        output = formatter.format_team_profile(
            sample_team_profile, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "Basic Info" in output
        assert "Recent Form" in output
        assert "World Cup History" in output
        assert "Advanced Stats" in output

    def test_contains_team_name_and_flag(
        self, formatter: OutputFormatter, sample_team_profile: TeamProfile
    ):
        output = formatter.format_team_profile(
            sample_team_profile, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "Brazil" in output
        assert "🇧🇷" in output
        assert "巴西" in output


# ============================================================================
# FOOTER TESTS
# ============================================================================


class TestFooter:
    """Tests for footer with data_updated_at and model_version."""

    def test_footer_present_in_match_prediction(
        self, formatter: OutputFormatter, sample_match_prediction: MatchPrediction
    ):
        output = formatter.format_match_prediction(
            sample_match_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "data_updated_at: 2026-06-10T14:30:00Z" in output
        assert f"model_version: {MODEL_VERSION}" in output

    def test_footer_present_in_group_standings(
        self, formatter: OutputFormatter, sample_group_prediction: GroupPrediction
    ):
        output = formatter.format_group_standings(
            sample_group_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "data_updated_at: 2026-06-10T14:30:00Z" in output
        assert f"model_version: {MODEL_VERSION}" in output

    def test_footer_present_in_champion_prediction(
        self, formatter: OutputFormatter, sample_champion_prediction: ChampionPrediction
    ):
        output = formatter.format_champion_prediction(
            sample_champion_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "data_updated_at: 2026-06-10T14:30:00Z" in output
        assert f"model_version: {MODEL_VERSION}" in output

    def test_footer_present_in_team_profile(
        self, formatter: OutputFormatter, sample_team_profile: TeamProfile
    ):
        output = formatter.format_team_profile(
            sample_team_profile, data_updated_at="2026-06-10T14:30:00Z"
        )
        assert "data_updated_at: 2026-06-10T14:30:00Z" in output
        assert f"model_version: {MODEL_VERSION}" in output

    def test_footer_iso8601_format(
        self, formatter: OutputFormatter, sample_match_prediction: MatchPrediction
    ):
        output = formatter.format_match_prediction(
            sample_match_prediction, data_updated_at="2026-06-10T14:30:00Z"
        )
        # ISO 8601 format check: YYYY-MM-DDTHH:MM:SSZ
        assert "2026-06-10T14:30:00Z" in output

    def test_footer_auto_generates_timestamp_when_none(
        self, formatter: OutputFormatter, sample_match_prediction: MatchPrediction
    ):
        output = formatter.format_match_prediction(sample_match_prediction)
        assert "data_updated_at:" in output
        assert f"model_version: {MODEL_VERSION}" in output
