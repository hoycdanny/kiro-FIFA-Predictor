"""
Unit tests for the Elo Rating Model.
"""

import math
import pytest

from src.data.data_manager import TeamProfile
from src.engine.elo_model import EloModel


def _make_team(name: str, elo_rating: int) -> TeamProfile:
    """Helper to create a minimal TeamProfile for Elo testing."""
    return TeamProfile(
        name=name,
        name_zh="",
        aliases=[],
        confederation="UEFA",
        fifa_ranking=1,
        fifa_points=1500.0,
        elo_rating=elo_rating,
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
    )


class TestEloModelProbabilitySum:
    """Test that probabilities always sum to 1.0."""

    def test_equal_teams_sum_to_one(self):
        model = EloModel()
        team_a = _make_team("TeamA", 1800)
        team_b = _make_team("TeamB", 1800)
        win_a, draw, win_b = model.predict(team_a, team_b)
        assert math.isclose(win_a + draw + win_b, 1.0, abs_tol=1e-9)

    def test_strong_vs_weak_sum_to_one(self):
        model = EloModel()
        team_a = _make_team("TeamA", 2100)
        team_b = _make_team("TeamB", 1500)
        win_a, draw, win_b = model.predict(team_a, team_b)
        assert math.isclose(win_a + draw + win_b, 1.0, abs_tol=1e-9)

    def test_weak_vs_strong_sum_to_one(self):
        model = EloModel()
        team_a = _make_team("TeamA", 1400)
        team_b = _make_team("TeamB", 2000)
        win_a, draw, win_b = model.predict(team_a, team_b)
        assert math.isclose(win_a + draw + win_b, 1.0, abs_tol=1e-9)

    def test_with_host_nation_bonus_sum_to_one(self):
        model = EloModel()
        team_a = _make_team("United States", 1800)
        team_b = _make_team("Brazil", 2000)
        win_a, draw, win_b = model.predict(team_a, team_b, venue_country="United States")
        assert math.isclose(win_a + draw + win_b, 1.0, abs_tol=1e-9)


class TestEloModelSymmetry:
    """Test symmetry properties of the Elo model."""

    def test_equal_teams_symmetric(self):
        model = EloModel()
        team_a = _make_team("TeamA", 1800)
        team_b = _make_team("TeamB", 1800)
        win_a, draw, win_b = model.predict(team_a, team_b)
        assert math.isclose(win_a, win_b, abs_tol=1e-9)

    def test_higher_elo_has_higher_win_prob(self):
        model = EloModel()
        team_a = _make_team("TeamA", 2000)
        team_b = _make_team("TeamB", 1700)
        win_a, draw, win_b = model.predict(team_a, team_b)
        assert win_a > win_b


class TestEloModelHostNationBonus:
    """Test host nation Elo bonus behavior."""

    def test_host_nation_bonus_boosts_team_a(self):
        model = EloModel()
        team_a = _make_team("United States", 1800)
        team_b = _make_team("Brazil", 1800)

        # Without bonus (neutral)
        win_a_neutral, _, _ = model.predict(team_a, team_b, venue_country="")

        # With bonus (USA playing at home)
        win_a_home, _, _ = model.predict(team_a, team_b, venue_country="United States")

        assert win_a_home > win_a_neutral

    def test_host_nation_bonus_boosts_team_b(self):
        model = EloModel()
        team_a = _make_team("Brazil", 1800)
        team_b = _make_team("Mexico", 1800)

        # Without bonus (neutral)
        _, _, win_b_neutral = model.predict(team_a, team_b, venue_country="")

        # With bonus (Mexico playing at home)
        _, _, win_b_home = model.predict(team_a, team_b, venue_country="Mexico")

        assert win_b_home > win_b_neutral

    def test_canada_host_bonus(self):
        model = EloModel()
        team_a = _make_team("Canada", 1700)
        team_b = _make_team("Germany", 1900)

        win_a_neutral, _, _ = model.predict(team_a, team_b, venue_country="")
        win_a_home, _, _ = model.predict(team_a, team_b, venue_country="Canada")

        assert win_a_home > win_a_neutral

    def test_non_host_no_bonus(self):
        model = EloModel()
        team_a = _make_team("Brazil", 1800)
        team_b = _make_team("Argentina", 1800)

        # Playing in USA venue but neither team is USA
        win_a_venue, draw_venue, win_b_venue = model.predict(
            team_a, team_b, venue_country="United States"
        )
        win_a_neutral, draw_neutral, win_b_neutral = model.predict(
            team_a, team_b, venue_country=""
        )

        assert math.isclose(win_a_venue, win_a_neutral, abs_tol=1e-9)
        assert math.isclose(draw_venue, draw_neutral, abs_tol=1e-9)
        assert math.isclose(win_b_venue, win_b_neutral, abs_tol=1e-9)

    def test_neutral_venue_no_advantage(self):
        model = EloModel()
        team_a = _make_team("United States", 1800)
        team_b = _make_team("Brazil", 1800)

        # Playing in a non-host country
        win_a, draw, win_b = model.predict(team_a, team_b, venue_country="Qatar")
        assert math.isclose(win_a, win_b, abs_tol=1e-9)


class TestEloModelNonNegative:
    """Test that all probabilities are non-negative."""

    def test_extreme_elo_difference(self):
        model = EloModel()
        team_a = _make_team("TeamA", 2400)
        team_b = _make_team("TeamB", 1200)
        win_a, draw, win_b = model.predict(team_a, team_b)
        assert win_a >= 0.0
        assert draw >= 0.0
        assert win_b >= 0.0

    def test_very_low_elo(self):
        model = EloModel()
        team_a = _make_team("TeamA", 1000)
        team_b = _make_team("TeamB", 1000)
        win_a, draw, win_b = model.predict(team_a, team_b)
        assert win_a >= 0.0
        assert draw >= 0.0
        assert win_b >= 0.0


class TestEloModelDrawProbability:
    """Test draw probability derivation."""

    def test_draw_highest_for_equal_teams(self):
        """Draw probability should be highest when teams are equal."""
        model = EloModel()
        team_equal_a = _make_team("TeamA", 1800)
        team_equal_b = _make_team("TeamB", 1800)
        _, draw_equal, _ = model.predict(team_equal_a, team_equal_b)

        team_strong = _make_team("Strong", 2100)
        team_weak = _make_team("Weak", 1500)
        _, draw_unequal, _ = model.predict(team_strong, team_weak)

        assert draw_equal > draw_unequal

    def test_draw_reasonable_range(self):
        """Draw probability for equal teams should be in reasonable WC range."""
        model = EloModel()
        team_a = _make_team("TeamA", 1800)
        team_b = _make_team("TeamB", 1800)
        _, draw, _ = model.predict(team_a, team_b)
        # World Cup draw rates typically around 20-30%
        assert 0.15 <= draw <= 0.40
