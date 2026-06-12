"""
Property-based tests for the Dynamic Factor Model and Elo Model.

**Validates: Requirements 7.2, 7.5, 7.6, 7.7**
"""

from datetime import date, timedelta

from hypothesis import given, settings, assume
from hypothesis.strategies import (
    booleans,
    composite,
    floats,
    integers,
    just,
    one_of,
    sampled_from,
)

from src.data.data_manager import TeamProfile
from src.engine.dynamic_factor import DynamicFactorModel
from src.engine.elo_model import EloModel


CONFEDERATIONS = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"]

TEAM_NAMES = [
    "Brazil", "Argentina", "France", "Germany", "England",
    "Spain", "Portugal", "Netherlands", "Italy", "Croatia",
]


def _make_team_profile(
    name: str = "TestTeam",
    elo_rating: int = 1500,
    current_win_streak: int = 0,
    current_loss_streak: int = 0,
    last_match_date: str | None = None,
    eliminated_by_2022: str | None = None,
    confederation: str = "UEFA",
) -> TeamProfile:
    """Helper to create a TeamProfile with given dynamic parameters."""
    return TeamProfile(
        name=name,
        name_zh="测试队",
        aliases=[],
        confederation=confederation,
        fifa_ranking=1,
        fifa_points=1500.0,
        elo_rating=elo_rating,
        group="A",
        recent_goals_avg=1.5,
        recent_conceded_avg=1.0,
        recent_win_rate=50.0,
        recent_draw_rate=25.0,
        recent_loss_rate=25.0,
        neutral_win_rate=50.0,
        best_wc_result="Group stage",
        vs_top20_win_rate=30.0,
        wc_first_match_win_rate=50.0,
        penalty_shootout_win_rate=50.0,
        first_half_goal_pct=45.0,
        second_half_goal_pct=55.0,
        clean_sheet_rate=20.0,
        failed_to_score_rate=15.0,
        current_win_streak=current_win_streak,
        current_loss_streak=current_loss_streak,
        last_match_date=last_match_date,
        eliminated_by_2022=eliminated_by_2022,
    )


class TestDynamicFactorCompositeCalculation:
    """
    Property 10: Dynamic factor composite calculation.

    For any team and opponent pair, the dynamic factor adjustment SHALL equal
    the sum of: +0.05 if win_streak >= 3, -0.05 if loss_streak >= 3,
    -0.03 if days_since_last_match < 3, and +0.03 if opponent is the team's
    2022 World Cup eliminator. Only applicable conditions contribute.

    **Validates: Requirements 7.5, 7.6, 7.7**
    """

    @given(
        win_streak=integers(min_value=0, max_value=10),
        loss_streak=integers(min_value=0, max_value=10),
        has_fatigue=booleans(),
        has_revenge=booleans(),
    )
    @settings(max_examples=200)
    def test_dynamic_factor_composite(
        self,
        win_streak: int,
        loss_streak: int,
        has_fatigue: bool,
        has_revenge: bool,
    ):
        """
        The dynamic factor adjustment equals the sum of applicable conditions.
        """
        model = DynamicFactorModel()
        match_date = date(2026, 6, 15)

        # Set up last_match_date based on fatigue flag
        if has_fatigue:
            # Less than 3 days rest -> fatigue
            last_match_date = (match_date - timedelta(days=2)).isoformat()
        else:
            # 3+ days rest -> no fatigue
            last_match_date = (match_date - timedelta(days=5)).isoformat()

        # Set up revenge opponent
        opponent_name = "France"
        eliminated_by = opponent_name if has_revenge else None

        team = _make_team_profile(
            name="Brazil",
            current_win_streak=win_streak,
            current_loss_streak=loss_streak,
            last_match_date=last_match_date,
            eliminated_by_2022=eliminated_by,
        )
        opponent = _make_team_profile(name=opponent_name)

        result = model.calculate_adjustment(team, opponent, match_date=match_date)

        # Calculate expected adjustment
        expected = 0.0
        if win_streak >= 3:
            expected += 0.05
        if loss_streak >= 3:
            expected -= 0.05
        if has_fatigue:
            expected += -0.03
        if has_revenge:
            expected += 0.03

        assert abs(result - expected) < 1e-9, (
            f"Expected {expected}, got {result}. "
            f"win_streak={win_streak}, loss_streak={loss_streak}, "
            f"fatigue={has_fatigue}, revenge={has_revenge}"
        )

    @given(
        win_streak=integers(min_value=3, max_value=10),
    )
    @settings(max_examples=50)
    def test_win_streak_contributes_bonus(self, win_streak: int):
        """Win streak >= 3 always contributes +0.05."""
        model = DynamicFactorModel()
        team = _make_team_profile(
            name="Brazil",
            current_win_streak=win_streak,
            current_loss_streak=0,
        )
        opponent = _make_team_profile(name="France")

        result = model.calculate_adjustment(team, opponent)
        assert result >= 0.05 - 1e-9, (
            f"Expected at least +0.05 with win_streak={win_streak}, got {result}"
        )

    @given(
        loss_streak=integers(min_value=3, max_value=10),
    )
    @settings(max_examples=50)
    def test_loss_streak_contributes_penalty(self, loss_streak: int):
        """Loss streak >= 3 always contributes -0.05."""
        model = DynamicFactorModel()
        team = _make_team_profile(
            name="Brazil",
            current_win_streak=0,
            current_loss_streak=loss_streak,
        )
        opponent = _make_team_profile(name="France")

        result = model.calculate_adjustment(team, opponent)
        assert result <= -0.05 + 1e-9, (
            f"Expected at most -0.05 with loss_streak={loss_streak}, got {result}"
        )


class TestEloModelProbabilityCompleteness:
    """
    Property 12: Elo model probability completeness.

    For any two valid Elo ratings and any home_advantage value,
    P(win_a) + P(draw) + P(win_b) SHALL equal 1.0 (within ±1e-9).

    **Validates: Requirements 7.2**
    """

    @given(
        elo_a=integers(min_value=800, max_value=2200),
        elo_b=integers(min_value=800, max_value=2200),
        venue_country=sampled_from(["", "United States", "Canada", "Mexico"]),
    )
    @settings(max_examples=200)
    def test_probabilities_sum_to_one(
        self, elo_a: int, elo_b: int, venue_country: str
    ):
        """P(win_a) + P(draw) + P(win_b) must equal 1.0."""
        model = EloModel()

        # Create team profiles with generated Elo ratings
        # Use host nation names when venue matches for host bonus testing
        team_a_name = "United States" if venue_country == "United States" else "Brazil"
        team_b_name = "Canada" if venue_country == "Canada" else "Germany"

        team_a = _make_team_profile(name=team_a_name, elo_rating=elo_a)
        team_b = _make_team_profile(name=team_b_name, elo_rating=elo_b)

        win_a, draw, win_b = model.predict(team_a, team_b, venue_country=venue_country)

        total = win_a + draw + win_b
        assert abs(total - 1.0) < 1e-9, (
            f"Probabilities sum to {total}, expected 1.0. "
            f"win_a={win_a}, draw={draw}, win_b={win_b}, "
            f"elo_a={elo_a}, elo_b={elo_b}, venue={venue_country}"
        )

    @given(
        elo_a=integers(min_value=800, max_value=2200),
        elo_b=integers(min_value=800, max_value=2200),
        venue_country=sampled_from(["", "United States", "Canada", "Mexico"]),
    )
    @settings(max_examples=200)
    def test_all_probabilities_non_negative(
        self, elo_a: int, elo_b: int, venue_country: str
    ):
        """All probabilities must be non-negative."""
        model = EloModel()

        team_a_name = "United States" if venue_country == "United States" else "Brazil"
        team_b_name = "Canada" if venue_country == "Canada" else "Germany"

        team_a = _make_team_profile(name=team_a_name, elo_rating=elo_a)
        team_b = _make_team_profile(name=team_b_name, elo_rating=elo_b)

        win_a, draw, win_b = model.predict(team_a, team_b, venue_country=venue_country)

        assert win_a >= 0.0, f"win_a is negative: {win_a}"
        assert draw >= 0.0, f"draw is negative: {draw}"
        assert win_b >= 0.0, f"win_b is negative: {win_b}"

    @given(
        elo_a=integers(min_value=800, max_value=2200),
        elo_b=integers(min_value=800, max_value=2200),
    )
    @settings(max_examples=100)
    def test_neutral_venue_completeness(self, elo_a: int, elo_b: int):
        """Neutral venue (no home advantage) still sums to 1.0."""
        model = EloModel()

        team_a = _make_team_profile(name="Brazil", elo_rating=elo_a)
        team_b = _make_team_profile(name="Germany", elo_rating=elo_b)

        win_a, draw, win_b = model.predict(team_a, team_b, venue_country="")

        total = win_a + draw + win_b
        assert abs(total - 1.0) < 1e-9, (
            f"Neutral venue probabilities sum to {total}, expected 1.0"
        )
