"""
Unit tests for the Prediction Engine.

Tests verify:
1. predict_match returns valid MatchPrediction with W+D+L summing to 1.0
2. over_2_5 + under_2_5 sums to 1.0
3. confidence_index is integer in [0, 100]
4. predict_group returns 4 ranked teams with consistent stats (W+D+L=3, points=3W+D)
"""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.data_manager import DataManager, TeamProfile
from src.engine.ensemble import EnsembleModel
from src.engine.prediction_engine import (
    GroupPrediction,
    GroupStanding,
    MatchPrediction,
    PredictionEngine,
)


# ============================================================================
# TEST HELPERS
# ============================================================================


def _make_team(
    name: str = "TeamA",
    name_zh: str = "球隊A",
    elo_rating: int = 1800,
    confederation: str = "UEFA",
    goals_avg: float = 1.5,
    conceded_avg: float = 1.0,
    group: str = "A",
    win_streak: int = 0,
    loss_streak: int = 0,
    last_match_date: str | None = None,
    eliminated_by_2022: str | None = None,
) -> TeamProfile:
    """Helper to create a minimal TeamProfile for testing."""
    return TeamProfile(
        name=name,
        name_zh=name_zh,
        aliases=[],
        confederation=confederation,
        fifa_ranking=10,
        fifa_points=1500.0,
        elo_rating=elo_rating,
        group=group,
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


def _make_data_manager_mock(teams: list[TeamProfile], groups: dict[str, list[str]]):
    """Create a mock DataManager with given teams and groups."""
    dm = MagicMock(spec=DataManager)
    dm.load_teams.return_value = teams
    dm.load_groups.return_value = groups
    return dm


def _make_engine_with_teams(
    teams: list[TeamProfile], groups: dict[str, list[str]]
) -> PredictionEngine:
    """Create a PredictionEngine with mock data."""
    dm = _make_data_manager_mock(teams, groups)
    ensemble = EnsembleModel()
    engine = PredictionEngine(data_manager=dm, ensemble=ensemble)
    return engine


# ============================================================================
# TEST PREDICT_MATCH - W+D+L Sum
# ============================================================================


class TestPredictMatchProbabilitySum:
    """Test that predict_match returns W+D+L summing to 1.0."""

    def _setup_engine(self) -> PredictionEngine:
        team_a = _make_team("Brazil", elo_rating=2000, goals_avg=2.0, conceded_avg=0.8)
        team_b = _make_team("Japan", elo_rating=1700, goals_avg=1.3, conceded_avg=1.2)
        teams = [team_a, team_b]
        groups = {"A": ["Brazil", "Japan", "TeamC", "TeamD"]}
        return _make_engine_with_teams(teams, groups)

    def test_wdl_sums_to_one(self):
        """W+D+L probabilities must sum to 1.0 (within tolerance)."""
        engine = self._setup_engine()
        pred = engine.predict_match("Brazil", "Japan")

        total = pred.win_prob + pred.draw_prob + pred.lose_prob
        assert math.isclose(total, 1.0, abs_tol=1e-6), (
            f"W+D+L = {total}, expected 1.0"
        )

    def test_wdl_non_negative(self):
        """All W/D/L probabilities must be non-negative."""
        engine = self._setup_engine()
        pred = engine.predict_match("Brazil", "Japan")

        assert pred.win_prob >= 0.0
        assert pred.draw_prob >= 0.0
        assert pred.lose_prob >= 0.0

    def test_wdl_with_analyst_style(self):
        """Analyst style should preserve W+D+L sum = 1.0."""
        engine = self._setup_engine()
        pred = engine.predict_match("Brazil", "Japan", coach_style="分析師")

        total = pred.win_prob + pred.draw_prob + pred.lose_prob
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_wdl_with_contrarian_style(self):
        """Contrarian style should preserve W+D+L sum = 1.0."""
        engine = self._setup_engine()
        pred = engine.predict_match("Brazil", "Japan", coach_style="反向思考者")

        total = pred.win_prob + pred.draw_prob + pred.lose_prob
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_wdl_with_tactician_style(self):
        """Tactician style should preserve W+D+L sum = 1.0."""
        engine = self._setup_engine()
        pred = engine.predict_match("Brazil", "Japan", coach_style="戰術家")

        total = pred.win_prob + pred.draw_prob + pred.lose_prob
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_stronger_team_has_higher_win(self):
        """Significantly stronger team should have higher win probability."""
        engine = self._setup_engine()
        pred = engine.predict_match("Brazil", "Japan")

        # Brazil has much higher Elo and goals avg
        assert pred.win_prob > pred.lose_prob


# ============================================================================
# TEST PREDICT_MATCH - Over/Under Sum
# ============================================================================


class TestPredictMatchOverUnder:
    """Test that over_2_5 + under_2_5 sums to 1.0."""

    def _setup_engine(self) -> PredictionEngine:
        team_a = _make_team("France", elo_rating=2050, goals_avg=2.2, conceded_avg=0.7)
        team_b = _make_team("Mexico", elo_rating=1750, goals_avg=1.4, conceded_avg=1.1)
        teams = [team_a, team_b]
        groups = {"A": ["France", "Mexico", "TeamC", "TeamD"]}
        return _make_engine_with_teams(teams, groups)

    def test_over_under_sums_to_one(self):
        """over_2_5 + under_2_5 must sum to 1.0."""
        engine = self._setup_engine()
        pred = engine.predict_match("France", "Mexico")

        total = pred.over_2_5 + pred.under_2_5
        assert math.isclose(total, 1.0, abs_tol=1e-6), (
            f"Over + Under = {total}, expected 1.0"
        )

    def test_over_under_non_negative(self):
        """Both over and under probabilities must be non-negative."""
        engine = self._setup_engine()
        pred = engine.predict_match("France", "Mexico")

        assert pred.over_2_5 >= 0.0
        assert pred.under_2_5 >= 0.0

    def test_over_under_within_bounds(self):
        """Both probabilities must be in [0, 1]."""
        engine = self._setup_engine()
        pred = engine.predict_match("France", "Mexico")

        assert 0.0 <= pred.over_2_5 <= 1.0
        assert 0.0 <= pred.under_2_5 <= 1.0


# ============================================================================
# TEST PREDICT_MATCH - Confidence Index
# ============================================================================


class TestPredictMatchConfidence:
    """Test that confidence_index is integer in [0, 100]."""

    def _setup_engine(self) -> PredictionEngine:
        team_a = _make_team("Germany", elo_rating=1950, goals_avg=1.9, conceded_avg=0.9)
        team_b = _make_team("Canada", elo_rating=1650, goals_avg=1.2, conceded_avg=1.3)
        teams = [team_a, team_b]
        groups = {"A": ["Germany", "Canada", "TeamC", "TeamD"]}
        return _make_engine_with_teams(teams, groups)

    def test_confidence_is_integer(self):
        """confidence_index must be an integer."""
        engine = self._setup_engine()
        pred = engine.predict_match("Germany", "Canada")

        assert isinstance(pred.confidence_index, int)

    def test_confidence_in_valid_range(self):
        """confidence_index must be in [0, 100]."""
        engine = self._setup_engine()
        pred = engine.predict_match("Germany", "Canada")

        assert 0 <= pred.confidence_index <= 100

    def test_confidence_higher_for_dominant_outcome(self):
        """When one outcome is very dominant, confidence should be higher."""
        # Create a strong vs weak matchup
        strong = _make_team("Strong", elo_rating=2200, goals_avg=3.0, conceded_avg=0.3)
        weak = _make_team("Weak", elo_rating=1200, goals_avg=0.5, conceded_avg=2.5)
        teams = [strong, weak]
        groups = {"A": ["Strong", "Weak", "TeamC", "TeamD"]}
        engine = _make_engine_with_teams(teams, groups)

        pred = engine.predict_match("Strong", "Weak")

        # Strong dominance should yield higher confidence
        assert pred.confidence_index > 30

    def test_confidence_lower_for_balanced_teams(self):
        """When teams are balanced, confidence should be lower."""
        team_a = _make_team("EqualA", elo_rating=1800, goals_avg=1.5, conceded_avg=1.0)
        team_b = _make_team("EqualB", elo_rating=1800, goals_avg=1.5, conceded_avg=1.0)
        teams = [team_a, team_b]
        groups = {"A": ["EqualA", "EqualB", "TeamC", "TeamD"]}
        engine = _make_engine_with_teams(teams, groups)

        pred = engine.predict_match("EqualA", "EqualB")

        # Balanced teams should yield lower confidence
        assert pred.confidence_index < 70


# ============================================================================
# TEST PREDICT_MATCH - Top Scores
# ============================================================================


class TestPredictMatchTopScores:
    """Test top_scores output."""

    def _setup_engine(self) -> PredictionEngine:
        team_a = _make_team("Spain", elo_rating=1950, goals_avg=2.0, conceded_avg=0.8)
        team_b = _make_team("Morocco", elo_rating=1750, goals_avg=1.3, conceded_avg=1.0)
        teams = [team_a, team_b]
        groups = {"A": ["Spain", "Morocco", "TeamC", "TeamD"]}
        return _make_engine_with_teams(teams, groups)

    def test_top_scores_has_three_entries(self):
        """top_scores should contain exactly 3 entries."""
        engine = self._setup_engine()
        pred = engine.predict_match("Spain", "Morocco")

        assert len(pred.top_scores) == 3

    def test_top_scores_format(self):
        """Each entry should be (int, int, float) tuple."""
        engine = self._setup_engine()
        pred = engine.predict_match("Spain", "Morocco")

        for score_a, score_b, prob in pred.top_scores:
            assert isinstance(score_a, int)
            assert isinstance(score_b, int)
            assert isinstance(prob, float)
            assert score_a >= 0
            assert score_b >= 0
            assert prob > 0

    def test_top_scores_sorted_by_probability(self):
        """Top scores should be sorted by probability descending."""
        engine = self._setup_engine()
        pred = engine.predict_match("Spain", "Morocco")

        probs = [p for _, _, p in pred.top_scores]
        assert probs == sorted(probs, reverse=True)


# ============================================================================
# TEST PREDICT_MATCH - Expected Goals
# ============================================================================


class TestPredictMatchExpectedGoals:
    """Test expected goals output."""

    def _setup_engine(self) -> PredictionEngine:
        team_a = _make_team("Argentina", elo_rating=2100, goals_avg=2.5, conceded_avg=0.6)
        team_b = _make_team("Nigeria", elo_rating=1600, goals_avg=1.0, conceded_avg=1.5)
        teams = [team_a, team_b]
        groups = {"A": ["Argentina", "Nigeria", "TeamC", "TeamD"]}
        return _make_engine_with_teams(teams, groups)

    def test_expected_goals_non_negative(self):
        """Expected goals should be non-negative."""
        engine = self._setup_engine()
        pred = engine.predict_match("Argentina", "Nigeria")

        assert pred.expected_goals_a >= 0.0
        assert pred.expected_goals_b >= 0.0

    def test_stronger_team_has_higher_xg(self):
        """Stronger team should have higher expected goals."""
        engine = self._setup_engine()
        pred = engine.predict_match("Argentina", "Nigeria")

        # Argentina is much stronger offensively
        assert pred.expected_goals_a > pred.expected_goals_b


# ============================================================================
# TEST PREDICT_GROUP
# ============================================================================


class TestPredictGroup:
    """Test predict_group returns valid group prediction."""

    def _setup_engine(self) -> PredictionEngine:
        teams = [
            _make_team("TeamA", elo_rating=2000, goals_avg=2.0, conceded_avg=0.8, group="A"),
            _make_team("TeamB", elo_rating=1800, goals_avg=1.5, conceded_avg=1.0, group="A"),
            _make_team("TeamC", elo_rating=1600, goals_avg=1.2, conceded_avg=1.3, group="A"),
            _make_team("TeamD", elo_rating=1400, goals_avg=0.8, conceded_avg=1.8, group="A"),
        ]
        groups = {"A": ["TeamA", "TeamB", "TeamC", "TeamD"]}
        return _make_engine_with_teams(teams, groups)

    def test_group_has_four_teams(self):
        """Group prediction should have exactly 4 teams in standings."""
        engine = self._setup_engine()
        result = engine.predict_group("A")

        assert len(result.standings) == 4

    def test_group_has_six_matches(self):
        """Group prediction should have exactly 6 match predictions."""
        engine = self._setup_engine()
        result = engine.predict_group("A")

        assert len(result.match_predictions) == 6

    def test_group_correct_group_id(self):
        """GroupPrediction should have correct group_id."""
        engine = self._setup_engine()
        result = engine.predict_group("A")

        assert result.group_id == "A"

    def test_group_each_team_played_three(self):
        """Each team should have played exactly 3 matches."""
        engine = self._setup_engine()
        result = engine.predict_group("A")

        for standing in result.standings:
            assert standing.played == 3

    def test_group_wdl_equals_three(self):
        """For each team: W + D + L = 3."""
        engine = self._setup_engine()
        result = engine.predict_group("A")

        for standing in result.standings:
            total_games = standing.wins + standing.draws + standing.losses
            assert total_games == 3, (
                f"Team {standing.team}: W+D+L = {total_games}, expected 3"
            )

    def test_group_points_formula(self):
        """For each team: points = 3*W + D."""
        engine = self._setup_engine()
        result = engine.predict_group("A")

        for standing in result.standings:
            expected_points = 3 * standing.wins + standing.draws
            assert standing.points == expected_points, (
                f"Team {standing.team}: points={standing.points}, "
                f"expected 3*{standing.wins}+{standing.draws}={expected_points}"
            )

    def test_group_goal_difference_formula(self):
        """For each team: GD = GF - GA."""
        engine = self._setup_engine()
        result = engine.predict_group("A")

        for standing in result.standings:
            expected_gd = standing.goals_for - standing.goals_against
            assert standing.goal_difference == expected_gd, (
                f"Team {standing.team}: GD={standing.goal_difference}, "
                f"expected {standing.goals_for}-{standing.goals_against}={expected_gd}"
            )

    def test_group_sorted_by_points(self):
        """Standings should be sorted by points descending."""
        engine = self._setup_engine()
        result = engine.predict_group("A")

        points = [s.points for s in result.standings]
        # Points should be in non-increasing order
        for i in range(len(points) - 1):
            assert points[i] >= points[i + 1]

    def test_group_top_two_qualified(self):
        """Top 2 teams should be marked as '確定晉級'."""
        engine = self._setup_engine()
        result = engine.predict_group("A")

        assert result.standings[0].qualification_status == "確定晉級"
        assert result.standings[1].qualification_status == "確定晉級"

    def test_group_third_has_qualification_probability(self):
        """Third-place team should have qualification probability."""
        engine = self._setup_engine()
        result = engine.predict_group("A")

        status = result.standings[2].qualification_status
        assert "可能晉級" in status

    def test_group_invalid_id_raises(self):
        """Invalid group ID should raise ValueError."""
        engine = self._setup_engine()

        with pytest.raises(ValueError):
            engine.predict_group("Z")

    def test_group_case_insensitive(self):
        """Group ID should be case-insensitive."""
        engine = self._setup_engine()

        result = engine.predict_group("a")
        assert result.group_id == "A"

    def test_group_total_goals_consistent(self):
        """Total goals scored across all teams should equal total goals conceded."""
        engine = self._setup_engine()
        result = engine.predict_group("A")

        total_gf = sum(s.goals_for for s in result.standings)
        total_ga = sum(s.goals_against for s in result.standings)
        assert total_gf == total_ga, (
            f"Total GF={total_gf} != Total GA={total_ga}"
        )


# ============================================================================
# TEST COACH STYLE INTEGRATION
# ============================================================================


class TestCoachStyleIntegration:
    """Test coach style applied through predict_match."""

    def _setup_engine(self) -> PredictionEngine:
        team_a = _make_team("England", elo_rating=1900, goals_avg=1.8, conceded_avg=0.9)
        team_b = _make_team("Jamaica", elo_rating=1400, goals_avg=0.8, conceded_avg=1.8)
        teams = [team_a, team_b]
        groups = {"A": ["England", "Jamaica", "TeamC", "TeamD"]}
        return _make_engine_with_teams(teams, groups)

    def test_coach_style_default_is_analyst(self):
        """Default (no style) should use analyst."""
        engine = self._setup_engine()
        pred = engine.predict_match("England", "Jamaica")

        assert pred.coach_style == "分析師"

    def test_coach_style_analyst_explicit(self):
        """Explicit analyst style."""
        engine = self._setup_engine()
        pred = engine.predict_match("England", "Jamaica", coach_style="分析師")

        assert pred.coach_style == "分析師"

    def test_coach_style_contrarian(self):
        """Contrarian style should be applied."""
        engine = self._setup_engine()
        pred = engine.predict_match("England", "Jamaica", coach_style="反向思考者")

        assert pred.coach_style == "反向思考者"
        total = pred.win_prob + pred.draw_prob + pred.lose_prob
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_coach_style_tactician(self):
        """Tactician style should be applied."""
        engine = self._setup_engine()
        pred = engine.predict_match("England", "Jamaica", coach_style="戰術家")

        assert pred.coach_style == "戰術家"
        total = pred.win_prob + pred.draw_prob + pred.lose_prob
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_coach_style_keyword_aggressive(self):
        """'aggressive' keyword should map to contrarian."""
        engine = self._setup_engine()
        pred = engine.predict_match("England", "Jamaica", coach_style="aggressive")

        assert pred.coach_style == "反向思考者"

    def test_coach_style_keyword_conservative(self):
        """'conservative' keyword should map to analyst."""
        engine = self._setup_engine()
        pred = engine.predict_match("England", "Jamaica", coach_style="conservative")

        assert pred.coach_style == "分析師"

    def test_coach_style_keyword_balanced(self):
        """'balanced' keyword should map to tactician."""
        engine = self._setup_engine()
        pred = engine.predict_match("England", "Jamaica", coach_style="balanced")

        assert pred.coach_style == "戰術家"


# ============================================================================
# TEST CONFIDENCE INDEX COMPUTATION
# ============================================================================


class TestConfidenceIndexComputation:
    """Test confidence index computation logic directly."""

    def test_equal_split_gives_zero(self):
        """Equal 33/33/33 split should give confidence near 0."""
        engine = PredictionEngine(
            data_manager=MagicMock(), ensemble=EnsembleModel()
        )
        confidence = engine._compute_confidence_index(1 / 3, 1 / 3, 1 / 3)
        assert confidence == 0

    def test_certainty_gives_hundred(self):
        """100% one outcome should give confidence 100."""
        engine = PredictionEngine(
            data_manager=MagicMock(), ensemble=EnsembleModel()
        )
        confidence = engine._compute_confidence_index(1.0, 0.0, 0.0)
        assert confidence == 100

    def test_moderate_dominance(self):
        """60/20/20 split should give moderate confidence."""
        engine = PredictionEngine(
            data_manager=MagicMock(), ensemble=EnsembleModel()
        )
        confidence = engine._compute_confidence_index(0.6, 0.2, 0.2)
        assert 20 < confidence < 60
