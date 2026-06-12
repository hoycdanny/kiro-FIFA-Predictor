"""Unit tests for the Dynamic Factor Model."""

from datetime import date

import pytest

from src.data.data_manager import TeamProfile
from src.engine.dynamic_factor import DynamicFactorModel, DynamicFactors


def _make_team(
    name: str = "TeamA",
    win_streak: int = 0,
    loss_streak: int = 0,
    last_match_date: str | None = None,
    eliminated_by_2022: str | None = None,
) -> TeamProfile:
    """Create a minimal TeamProfile for testing."""
    return TeamProfile(
        name=name,
        name_zh="測試隊",
        aliases=[],
        confederation="UEFA",
        fifa_ranking=10,
        fifa_points=1500.0,
        elo_rating=1800,
        group="A",
        recent_goals_avg=1.5,
        recent_conceded_avg=0.8,
        recent_win_rate=60.0,
        recent_draw_rate=20.0,
        recent_loss_rate=20.0,
        neutral_win_rate=55.0,
        best_wc_result="Quarter-finals",
        vs_top20_win_rate=40.0,
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


class TestDynamicFactorModel:
    """Tests for DynamicFactorModel.calculate_adjustment()."""

    def setup_method(self):
        self.model = DynamicFactorModel()

    def test_no_conditions_returns_zero(self):
        """No applicable conditions → adjustment = 0.0."""
        team = _make_team(win_streak=0, loss_streak=0)
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == 0.0

    def test_win_streak_bonus(self):
        """Win streak >= 3 → +0.05 adjustment."""
        team = _make_team(win_streak=3)
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == pytest.approx(0.05)

    def test_win_streak_above_threshold(self):
        """Win streak > 3 still gives +0.05."""
        team = _make_team(win_streak=5)
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == pytest.approx(0.05)

    def test_loss_streak_penalty(self):
        """Loss streak >= 3 → -0.05 adjustment."""
        team = _make_team(loss_streak=3)
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == pytest.approx(-0.05)

    def test_fatigue_penalty(self):
        """Rest < 3 days → -0.03 adjustment."""
        # Match on June 15, last match on June 13 (2 days rest)
        team = _make_team(last_match_date="2026-06-13")
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == pytest.approx(-0.03)

    def test_no_fatigue_at_threshold(self):
        """Rest == 3 days → no fatigue penalty."""
        # Match on June 15, last match on June 12 (3 days rest)
        team = _make_team(last_match_date="2026-06-12")
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == 0.0

    def test_no_fatigue_when_last_match_none(self):
        """last_match_date is None → no fatigue penalty."""
        team = _make_team(last_match_date=None)
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == 0.0

    def test_revenge_bonus(self):
        """Facing 2022 eliminator → +0.03 adjustment."""
        team = _make_team(eliminated_by_2022="TeamB")
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == pytest.approx(0.03)

    def test_no_revenge_against_different_opponent(self):
        """Facing team that is NOT 2022 eliminator → no revenge bonus."""
        team = _make_team(eliminated_by_2022="TeamC")
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == 0.0

    def test_multiple_conditions_stack(self):
        """Win streak + revenge = +0.05 + 0.03 = +0.08."""
        team = _make_team(win_streak=4, eliminated_by_2022="TeamB")
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == pytest.approx(0.08)

    def test_all_negative_conditions_stack(self):
        """Loss streak + fatigue = -0.05 + (-0.03) = -0.08."""
        team = _make_team(loss_streak=3, last_match_date="2026-06-14")
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == pytest.approx(-0.08)

    def test_win_streak_below_threshold_no_bonus(self):
        """Win streak < 3 → no bonus."""
        team = _make_team(win_streak=2)
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == 0.0

    def test_loss_streak_below_threshold_no_penalty(self):
        """Loss streak < 3 → no penalty."""
        team = _make_team(loss_streak=2)
        opponent = _make_team(name="TeamB")
        match_date = date(2026, 6, 15)

        result = self.model.calculate_adjustment(team, opponent, match_date)

        assert result == 0.0


class TestDynamicFactors:
    """Tests for the DynamicFactors dataclass."""

    def test_defaults(self):
        """Default values should be zero/neutral."""
        factors = DynamicFactors()
        assert factors.win_streak == 0
        assert factors.loss_streak == 0
        assert factors.days_since_last_match == 7
        assert factors.revenge_opponent is None

    def test_custom_values(self):
        """Can set custom values."""
        factors = DynamicFactors(
            win_streak=5,
            loss_streak=0,
            days_since_last_match=2,
            revenge_opponent="France",
        )
        assert factors.win_streak == 5
        assert factors.days_since_last_match == 2
        assert factors.revenge_opponent == "France"
