"""Unit tests for the Historical H2H Model."""

from dataclasses import dataclass
from typing import Optional

import pytest

from src.engine.h2h_model import H2HModel


@dataclass
class _FakeTeamProfile:
    """Minimal stand-in for TeamProfile used in tests."""

    name: str
    name_zh: str = ""
    aliases: list = None  # type: ignore[assignment]
    confederation: str = ""
    fifa_ranking: int = 1
    fifa_points: float = 0.0
    elo_rating: int = 1500
    group: str = "A"
    recent_goals_avg: float = 1.0
    recent_conceded_avg: float = 1.0
    recent_win_rate: float = 0.5
    recent_draw_rate: float = 0.25
    recent_loss_rate: float = 0.25
    neutral_win_rate: float = 0.5
    best_wc_result: str = ""
    vs_top20_win_rate: float = 0.0
    wc_first_match_win_rate: float = 0.0
    penalty_shootout_win_rate: float = 0.0
    first_half_goal_pct: float = 0.5
    second_half_goal_pct: float = 0.5
    clean_sheet_rate: float = 0.0
    failed_to_score_rate: float = 0.0
    current_win_streak: int = 0
    current_loss_streak: int = 0
    last_match_date: Optional[str] = None
    eliminated_by_2022: Optional[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class TestH2HModelNeutralFallback:
    """Tests for neutral fallback when no H2H data exists."""

    def test_no_data_returns_neutral_split(self):
        """No H2H data returns the neutral 33/34/33 split."""
        model = H2HModel()
        team_a = _FakeTeamProfile(name="Brazil")
        team_b = _FakeTeamProfile(name="Germany")

        win_a, draw, win_b = model.predict(team_a, team_b)

        assert win_a == 0.33
        assert draw == 0.34
        assert win_b == 0.33

    def test_empty_dict_returns_neutral_split(self):
        """Explicitly empty H2H dict returns neutral split."""
        model = H2HModel(h2h_data={})
        team_a = _FakeTeamProfile(name="Argentina")
        team_b = _FakeTeamProfile(name="France")

        win_a, draw, win_b = model.predict(team_a, team_b)

        assert win_a == 0.33
        assert draw == 0.34
        assert win_b == 0.33

    def test_pair_not_in_data_returns_neutral(self):
        """Teams not in the H2H data get the neutral split."""
        h2h_data = {
            ("Brazil", "Germany"): {"wins_a": 5, "draws": 3, "wins_b": 4}
        }
        model = H2HModel(h2h_data=h2h_data)
        team_a = _FakeTeamProfile(name="Spain")
        team_b = _FakeTeamProfile(name="Portugal")

        win_a, draw, win_b = model.predict(team_a, team_b)

        assert win_a == 0.33
        assert draw == 0.34
        assert win_b == 0.33

    def test_zero_total_matches_returns_neutral(self):
        """Record with all zeros returns neutral split."""
        h2h_data = {
            ("Brazil", "Germany"): {"wins_a": 0, "draws": 0, "wins_b": 0}
        }
        model = H2HModel(h2h_data=h2h_data)
        team_a = _FakeTeamProfile(name="Brazil")
        team_b = _FakeTeamProfile(name="Germany")

        win_a, draw, win_b = model.predict(team_a, team_b)

        assert win_a == 0.33
        assert draw == 0.34
        assert win_b == 0.33


class TestH2HModelProbabilitySums:
    """Tests verifying probabilities always sum to 1.0."""

    def test_neutral_split_sums_to_one(self):
        """Neutral split sums to 1.0."""
        model = H2HModel()
        team_a = _FakeTeamProfile(name="Brazil")
        team_b = _FakeTeamProfile(name="Germany")

        probs = model.predict(team_a, team_b)

        assert abs(sum(probs) - 1.0) < 1e-9

    def test_historical_data_sums_to_one(self):
        """Probabilities from historical data sum to 1.0."""
        h2h_data = {
            ("Brazil", "Germany"): {"wins_a": 5, "draws": 3, "wins_b": 4}
        }
        model = H2HModel(h2h_data=h2h_data)
        team_a = _FakeTeamProfile(name="Brazil")
        team_b = _FakeTeamProfile(name="Germany")

        probs = model.predict(team_a, team_b)

        assert abs(sum(probs) - 1.0) < 1e-9

    def test_single_match_record_sums_to_one(self):
        """Single match record still sums to 1.0."""
        h2h_data = {
            ("Japan", "Colombia"): {"wins_a": 1, "draws": 0, "wins_b": 0}
        }
        model = H2HModel(h2h_data=h2h_data)
        team_a = _FakeTeamProfile(name="Japan")
        team_b = _FakeTeamProfile(name="Colombia")

        probs = model.predict(team_a, team_b)

        assert abs(sum(probs) - 1.0) < 1e-9


class TestH2HModelHistoricalProbabilities:
    """Tests verifying probabilities reflect historical record."""

    def test_dominant_team_gets_higher_probability(self):
        """Team with more wins gets higher win probability."""
        h2h_data = {
            ("Brazil", "Germany"): {"wins_a": 8, "draws": 2, "wins_b": 2}
        }
        model = H2HModel(h2h_data=h2h_data)
        team_a = _FakeTeamProfile(name="Brazil")
        team_b = _FakeTeamProfile(name="Germany")

        win_a, draw, win_b = model.predict(team_a, team_b)

        assert win_a > win_b
        assert win_a == pytest.approx(8 / 12)
        assert draw == pytest.approx(2 / 12)
        assert win_b == pytest.approx(2 / 12)

    def test_even_record_gives_equal_probabilities(self):
        """Even record gives equal win probabilities."""
        h2h_data = {
            ("Spain", "Italy"): {"wins_a": 5, "draws": 5, "wins_b": 5}
        }
        model = H2HModel(h2h_data=h2h_data)
        team_a = _FakeTeamProfile(name="Spain")
        team_b = _FakeTeamProfile(name="Italy")

        win_a, draw, win_b = model.predict(team_a, team_b)

        assert win_a == pytest.approx(1 / 3)
        assert draw == pytest.approx(1 / 3)
        assert win_b == pytest.approx(1 / 3)

    def test_all_draws_gives_draw_probability_one(self):
        """All draws gives draw probability of 1.0."""
        h2h_data = {
            ("France", "England"): {"wins_a": 0, "draws": 10, "wins_b": 0}
        }
        model = H2HModel(h2h_data=h2h_data)
        team_a = _FakeTeamProfile(name="France")
        team_b = _FakeTeamProfile(name="England")

        win_a, draw, win_b = model.predict(team_a, team_b)

        assert win_a == 0.0
        assert draw == 1.0
        assert win_b == 0.0

    def test_reverse_lookup_works(self):
        """Looking up teams in reverse order still works correctly."""
        h2h_data = {
            ("Brazil", "Germany"): {"wins_a": 7, "draws": 2, "wins_b": 3}
        }
        model = H2HModel(h2h_data=h2h_data)
        # Query with Germany as team_a (reverse of stored key)
        team_a = _FakeTeamProfile(name="Germany")
        team_b = _FakeTeamProfile(name="Brazil")

        win_a, draw, win_b = model.predict(team_a, team_b)

        # Germany's wins should be Brazil's wins_b from stored record
        assert win_a == pytest.approx(3 / 12)  # Germany's wins
        assert draw == pytest.approx(2 / 12)
        assert win_b == pytest.approx(7 / 12)  # Brazil's wins
