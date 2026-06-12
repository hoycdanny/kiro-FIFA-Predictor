"""
Dixon-Coles Poisson Model for FIFA match prediction.

Implements the Dixon-Coles modification of the independent Poisson model
to produce a 5×5 score probability matrix with low-score corrections.
"""

import math

import numpy as np
from numpy.typing import NDArray

from src.data.data_manager import TeamProfile
from src.utils.constants import (
    CONFEDERATION_COEFFICIENTS,
    DIXON_COLES_RHO,
    LEAGUE_AVG_GOALS,
    NEUTRAL_VENUE_FACTOR,
    SCORE_MATRIX_SIZE,
)


class DixonColesModel:
    """
    Dixon-Coles Poisson model for predicting football match scores.

    Calculates expected goals (lambda) for each team using attack strength
    and defense weakness relative to league averages, then produces a
    score probability matrix with tau corrections for low-scoring outcomes.
    """

    def predict(self, team_a: TeamProfile, team_b: TeamProfile) -> NDArray[np.float64]:
        """
        Calculate 5×5 score probability matrix.

        attack_strength = (team_goals_avg / league_avg) × confederation_coeff
        defense_weakness = (opponent_conceded_avg / league_avg)
        lambda = attack_strength × defense_weakness × neutral_factor

        Args:
            team_a: Profile of team A (home/first listed).
            team_b: Profile of team B (away/second listed).

        Returns:
            NDArray of shape (5, 5) where entry [i][j] is the probability
            of team A scoring i goals and team B scoring j goals.
        """
        # Calculate attack strength for each team
        conf_coeff_a = CONFEDERATION_COEFFICIENTS.get(team_a.confederation, 1.0)
        conf_coeff_b = CONFEDERATION_COEFFICIENTS.get(team_b.confederation, 1.0)

        attack_strength_a = (team_a.recent_goals_avg / LEAGUE_AVG_GOALS) * conf_coeff_a
        attack_strength_b = (team_b.recent_goals_avg / LEAGUE_AVG_GOALS) * conf_coeff_b

        # Calculate defense weakness for each team
        defense_weakness_a = team_b.recent_conceded_avg / LEAGUE_AVG_GOALS
        defense_weakness_b = team_a.recent_conceded_avg / LEAGUE_AVG_GOALS

        # Calculate expected goals (lambda) for each team
        lambda_a = attack_strength_a * defense_weakness_a * NEUTRAL_VENUE_FACTOR
        lambda_b = attack_strength_b * defense_weakness_b * NEUTRAL_VENUE_FACTOR

        # Build the score probability matrix
        matrix = np.zeros((SCORE_MATRIX_SIZE, SCORE_MATRIX_SIZE), dtype=np.float64)

        for i in range(SCORE_MATRIX_SIZE):
            for j in range(SCORE_MATRIX_SIZE):
                prob_i = self._poisson_probability(i, lambda_a)
                prob_j = self._poisson_probability(j, lambda_b)
                tau = self._tau_correction(i, j, lambda_a, lambda_b, DIXON_COLES_RHO)
                matrix[i, j] = prob_i * prob_j * tau

        return matrix

    def _tau_correction(
        self,
        goals_a: int,
        goals_b: int,
        lambda_a: float,
        lambda_b: float,
        rho: float,
    ) -> float:
        """
        Dixon-Coles tau correction for low-scoring outcomes.

        Adjusts probabilities for 0-0, 1-0, 0-1, and 1-1 scores to account
        for the observed correlation between low-scoring teams.

        Args:
            goals_a: Goals scored by team A.
            goals_b: Goals scored by team B.
            lambda_a: Expected goals for team A.
            lambda_b: Expected goals for team B.
            rho: Correction parameter (typically -0.1 to -0.2).

        Returns:
            Tau correction factor (multiplicative, clamped to non-negative).
        """
        if goals_a == 0 and goals_b == 0:
            tau = 1 - lambda_a * lambda_b * rho
        elif goals_a == 1 and goals_b == 0:
            tau = 1 + lambda_b * rho
        elif goals_a == 0 and goals_b == 1:
            tau = 1 + lambda_a * rho
        elif goals_a == 1 and goals_b == 1:
            tau = 1 - rho
        else:
            tau = 1.0
        # Clamp to non-negative to ensure valid probabilities
        return max(0.0, tau)

    def _poisson_probability(self, k: int, lam: float) -> float:
        """
        Calculate Poisson probability P(X=k) = (e^-λ × λ^k) / k!

        Args:
            k: Number of goals (non-negative integer).
            lam: Expected number of goals (lambda, positive float).

        Returns:
            Probability of exactly k goals given expected rate lam.
        """
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
