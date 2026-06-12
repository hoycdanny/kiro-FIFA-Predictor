"""
Dynamic Factor Model for the FIFA Predictor Power.

Calculates adjustments based on team momentum, fatigue, and revenge factors.
Only applicable conditions contribute to the final adjustment.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from src.data.data_manager import TeamProfile
from src.utils.constants import (
    FATIGUE_DAYS_THRESHOLD,
    FATIGUE_PENALTY,
    REVENGE_BONUS,
    STREAK_BONUS,
    STREAK_THRESHOLD,
)


@dataclass
class DynamicFactors:
    """Team's current dynamic factors."""

    win_streak: int = 0  # Consecutive wins
    loss_streak: int = 0  # Consecutive losses
    days_since_last_match: int = 7
    revenge_opponent: Optional[str] = None  # 2022 eliminator


class DynamicFactorModel:
    """
    Calculates dynamic factor adjustments based on team form and context.

    Factors:
    - Win streak >= 3: +5% bonus
    - Loss streak >= 3: -5% penalty
    - Rest < 3 days: -3% fatigue penalty
    - Facing 2022 eliminator: +3% revenge bonus
    """

    STREAK_BONUS: float = STREAK_BONUS
    FATIGUE_PENALTY: float = FATIGUE_PENALTY
    REVENGE_BONUS: float = REVENGE_BONUS
    STREAK_THRESHOLD: int = STREAK_THRESHOLD

    def calculate_adjustment(
        self,
        team: TeamProfile,
        opponent: TeamProfile,
        match_date: Optional[date] = None,
    ) -> float:
        """
        Calculate dynamic factor adjustment for a team.

        Args:
            team: The team to calculate adjustment for.
            opponent: The opposing team.
            match_date: The date of the match (defaults to today if not provided).

        Returns:
            Adjustment ratio (e.g., +0.05 means +5%).
            Only applicable conditions contribute.
        """
        adjustment = 0.0

        # Win streak bonus: +5% if win_streak >= 3
        if team.current_win_streak >= self.STREAK_THRESHOLD:
            adjustment += self.STREAK_BONUS

        # Loss streak penalty: -5% if loss_streak >= 3
        if team.current_loss_streak >= self.STREAK_THRESHOLD:
            adjustment -= self.STREAK_BONUS

        # Fatigue penalty: -3% if rest < 3 days
        if team.last_match_date is not None:
            if match_date is None:
                match_date = date.today()
            last_match = datetime.fromisoformat(team.last_match_date).date()
            days_rest = (match_date - last_match).days
            if days_rest < FATIGUE_DAYS_THRESHOLD:
                adjustment += self.FATIGUE_PENALTY

        # Revenge bonus: +3% if facing 2022 eliminator
        if (
            team.eliminated_by_2022 is not None
            and team.eliminated_by_2022 == opponent.name
        ):
            adjustment += self.REVENGE_BONUS

        return adjustment
