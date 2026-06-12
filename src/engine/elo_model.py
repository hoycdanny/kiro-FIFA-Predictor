"""
Elo Rating Model for the FIFA Predictor Power.

Calculates win/draw/lose probabilities based on Elo ratings,
with host nation bonus for USA, Canada, and Mexico when playing
in their country.
"""

import math

from src.data.data_manager import TeamProfile
from src.utils.constants import HOST_NATIONS, HOST_NATION_ELO_BONUS


class EloModel:
    """
    Elo-based match outcome predictor.

    Uses the standard Elo expected score formula:
        P(A) = 1 / (1 + 10^((Elo_B - Elo_A + home_advantage) / 400))

    Draw probability is derived using a factor based on how close
    the expected scores are (geometric mean approach scaled for
    international football draw rates ~22%).
    """

    HOME_ADVANTAGE_NEUTRAL: int = 0
    HOST_NATION_BONUS: int = HOST_NATION_ELO_BONUS
    DRAW_CONSTANT: float = 0.80  # Tuned for ~22% WC draw rate

    def predict(
        self, team_a: TeamProfile, team_b: TeamProfile, venue_country: str = ""
    ) -> tuple[float, float, float]:
        """
        Calculate Elo win/draw/lose probabilities.

        P(A) = 1 / (1 + 10^((Elo_B - Elo_A + home_advantage) / 400))

        Returns (win_a, draw, win_b) where all values are in [0, 1]
        and sum to 1.0.
        """
        home_advantage = self._get_home_advantage(team_a, team_b, venue_country)

        # Calculate expected score for team A using Elo formula
        elo_diff = team_b.elo_rating - team_a.elo_rating + home_advantage
        e_a = 1.0 / (1.0 + math.pow(10, elo_diff / 400.0))
        e_b = 1.0 - e_a

        # Derive draw probability using geometric mean approach
        # P(draw) = c * sqrt(e_a * e_b) where c is calibrated for WC draw rates
        draw_prob = self.DRAW_CONSTANT * math.sqrt(e_a * e_b)

        # Derive win/lose from expected scores minus draw share
        win_a = e_a - draw_prob / 2.0
        win_b = e_b - draw_prob / 2.0

        # Ensure all probabilities are non-negative
        win_a = max(0.0, win_a)
        win_b = max(0.0, win_b)
        draw_prob = max(0.0, draw_prob)

        # Normalize to ensure sum = 1.0
        total = win_a + draw_prob + win_b
        if total > 0:
            win_a /= total
            draw_prob /= total
            win_b /= total
        else:
            # Fallback: equal split (should never happen with valid Elo)
            win_a = 1.0 / 3.0
            draw_prob = 1.0 / 3.0
            win_b = 1.0 / 3.0

        return (win_a, draw_prob, win_b)

    def _get_home_advantage(
        self, team_a: TeamProfile, team_b: TeamProfile, venue_country: str
    ) -> int:
        """
        Calculate home advantage adjustment to Elo difference.

        - Neutral venue (default): 0
        - Host nation (USA, Canada, Mexico) playing in their country:
          +50 bonus for that team (subtracted from Elo diff if team_a is host,
          added if team_b is host)

        The home advantage is applied as a modifier to (Elo_B - Elo_A),
        so a negative value benefits team A and a positive value benefits team B.
        """
        if not venue_country:
            return self.HOME_ADVANTAGE_NEUTRAL

        # Check if team A is a host nation playing at home
        if team_a.name in HOST_NATIONS and team_a.name == venue_country:
            return -self.HOST_NATION_BONUS  # Benefits team A (reduces Elo_B - Elo_A)

        # Check if team B is a host nation playing at home
        if team_b.name in HOST_NATIONS and team_b.name == venue_country:
            return self.HOST_NATION_BONUS  # Benefits team B (increases Elo_B - Elo_A)

        return self.HOME_ADVANTAGE_NEUTRAL
