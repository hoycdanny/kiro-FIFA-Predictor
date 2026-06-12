"""
Ensemble Model for the FIFA Predictor Power.

Combines predictions from multiple sub-models (Dixon-Coles, Elo, H2H,
Dynamic Factor) using weighted averaging with fallback handling for
sub-model failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.data.data_manager import TeamProfile
from src.engine.dixon_coles import DixonColesModel
from src.engine.dynamic_factor import DynamicFactorModel
from src.engine.elo_model import EloModel
from src.engine.h2h_model import H2HModel
from src.utils.constants import DEFAULT_WEIGHTS, WEIGHT_MAX, WEIGHT_MIN


class AllModelsFailedError(Exception):
    """Raised when all sub-models fail during prediction."""

    pass


@dataclass
class EnsembleWeights:
    """Ensemble model weights for combining sub-model predictions.

    Default weights: Poisson 0.40, Elo 0.25, H2H 0.15, Dynamic 0.20.
    Each weight must be in [WEIGHT_MIN, WEIGHT_MAX] and all must sum to 1.00.
    """

    poisson: float = 0.40
    elo: float = 0.25
    h2h: float = 0.15
    dynamic: float = 0.20

    def validate(self) -> bool:
        """Verify all weights in [0.10, 0.60] and sum = 1.00.

        Returns:
            True if all weights are within valid range and sum to 1.00,
            False otherwise.
        """
        weights = [self.poisson, self.elo, self.h2h, self.dynamic]
        return (
            all(WEIGHT_MIN <= w <= WEIGHT_MAX for w in weights)
            and abs(sum(weights) - 1.00) < 1e-9
        )

    def redistribute_without(self, excluded: str) -> "EnsembleWeights":
        """Exclude a sub-model, redistribute proportionally among remaining.

        The excluded model's weight is distributed proportionally to the
        remaining models based on their current weights.

        Args:
            excluded: Name of the model to exclude. Must be one of:
                "poisson", "elo", "h2h", "dynamic".

        Returns:
            New EnsembleWeights with the excluded model set to 0.0
            and remaining weights redistributed proportionally summing to 1.0.

        Raises:
            ValueError: If excluded model name is not recognized.
        """
        weight_map = {
            "poisson": self.poisson,
            "elo": self.elo,
            "h2h": self.h2h,
            "dynamic": self.dynamic,
        }

        if excluded not in weight_map:
            raise ValueError(
                f"Unknown model '{excluded}'. Must be one of: {list(weight_map.keys())}"
            )

        remaining = {k: v for k, v in weight_map.items() if k != excluded}
        remaining_sum = sum(remaining.values())

        if remaining_sum == 0:
            # Edge case: all remaining weights are zero (shouldn't happen with valid weights)
            n = len(remaining)
            redistributed = {k: 1.0 / n for k in remaining}
        else:
            redistributed = {k: v / remaining_sum for k, v in remaining.items()}

        return EnsembleWeights(
            poisson=redistributed.get("poisson", 0.0),
            elo=redistributed.get("elo", 0.0),
            h2h=redistributed.get("h2h", 0.0),
            dynamic=redistributed.get("dynamic", 0.0),
        )


class EnsembleModel:
    """Ensemble model combining multiple sub-model predictions.

    Integrates Dixon-Coles Poisson model, Elo rating model, historical H2H
    model, and dynamic factor model using weighted combination.
    """

    def __init__(
        self,
        weights: Optional[EnsembleWeights] = None,
        dixon_coles: Optional[DixonColesModel] = None,
        elo_model: Optional[EloModel] = None,
        h2h_model: Optional[H2HModel] = None,
        dynamic_factor: Optional[DynamicFactorModel] = None,
    ):
        """Initialize ensemble with weights and sub-models.

        Args:
            weights: Ensemble weights. Defaults to EnsembleWeights() if None.
            dixon_coles: Dixon-Coles Poisson model instance.
            elo_model: Elo rating model instance.
            h2h_model: H2H historical model instance.
            dynamic_factor: Dynamic factor model instance.
        """
        self.weights = weights or EnsembleWeights()
        self.dixon_coles = dixon_coles or DixonColesModel()
        self.elo_model = elo_model or EloModel()
        self.h2h_model = h2h_model or H2HModel()
        self.dynamic_factor = dynamic_factor or DynamicFactorModel()

    def predict(
        self, team_a: TeamProfile, team_b: TeamProfile
    ) -> tuple[float, float, float]:
        """Combine all sub-models using weighted average.

        Each sub-model contributes a (win_a, draw, win_b) prediction:
        - Dixon-Coles: 5×5 matrix → aggregate to (win_a, draw, win_b)
        - Elo: returns (win_a, draw, win_b) directly
        - H2H: returns (win_a, draw, win_b) directly
        - Dynamic factor: adjustment applied as multiplier to win probability

        Args:
            team_a: Profile of team A.
            team_b: Profile of team B.

        Returns:
            Tuple of (win_a, draw, win_b) probabilities summing to 1.0.
        """
        # Dixon-Coles: convert 5×5 matrix to (win_a, draw, win_b)
        matrix = self.dixon_coles.predict(team_a, team_b)
        dc_probs = self._matrix_to_probabilities(matrix)

        # Elo model: returns (win_a, draw, win_b) directly
        elo_probs = self.elo_model.predict(team_a, team_b)

        # H2H model: returns (win_a, draw, win_b) directly
        h2h_probs = self.h2h_model.predict(team_a, team_b)

        # Dynamic factor: calculate adjustment and apply to base probability
        adj_a = self.dynamic_factor.calculate_adjustment(team_a, team_b)
        adj_b = self.dynamic_factor.calculate_adjustment(team_b, team_a)
        dynamic_probs = self._apply_dynamic_adjustment(dc_probs, adj_a, adj_b)

        # Weighted combination
        win_a = (
            self.weights.poisson * dc_probs[0]
            + self.weights.elo * elo_probs[0]
            + self.weights.h2h * h2h_probs[0]
            + self.weights.dynamic * dynamic_probs[0]
        )
        draw = (
            self.weights.poisson * dc_probs[1]
            + self.weights.elo * elo_probs[1]
            + self.weights.h2h * h2h_probs[1]
            + self.weights.dynamic * dynamic_probs[1]
        )
        win_b = (
            self.weights.poisson * dc_probs[2]
            + self.weights.elo * elo_probs[2]
            + self.weights.h2h * h2h_probs[2]
            + self.weights.dynamic * dynamic_probs[2]
        )

        # Normalize to ensure sum = 1.0
        total = win_a + draw + win_b
        if total > 0:
            win_a /= total
            draw /= total
            win_b /= total
        else:
            win_a = 1.0 / 3.0
            draw = 1.0 / 3.0
            win_b = 1.0 / 3.0

        return (win_a, draw, win_b)

    def predict_with_fallback(
        self, team_a: TeamProfile, team_b: TeamProfile
    ) -> tuple[float, float, float]:
        """Try all sub-models; on failure, exclude failed model and redistribute.

        Attempts each sub-model prediction. If any model raises an exception,
        it is excluded and weights are redistributed proportionally among the
        remaining models. Records which models failed.

        At least one model must succeed. If all models fail, raises
        AllModelsFailedError.

        Args:
            team_a: Profile of team A.
            team_b: Profile of team B.

        Returns:
            Tuple of (win_a, draw, win_b) probabilities summing to 1.0.

        Raises:
            AllModelsFailedError: If all sub-models fail.
        """
        results: dict[str, tuple[float, float, float]] = {}
        failed_models: list[str] = []

        # Try Dixon-Coles
        try:
            matrix = self.dixon_coles.predict(team_a, team_b)
            results["poisson"] = self._matrix_to_probabilities(matrix)
        except Exception:
            failed_models.append("poisson")

        # Try Elo
        try:
            results["elo"] = self.elo_model.predict(team_a, team_b)
        except Exception:
            failed_models.append("elo")

        # Try H2H
        try:
            results["h2h"] = self.h2h_model.predict(team_a, team_b)
        except Exception:
            failed_models.append("h2h")

        # Try Dynamic Factor
        try:
            adj_a = self.dynamic_factor.calculate_adjustment(team_a, team_b)
            adj_b = self.dynamic_factor.calculate_adjustment(team_b, team_a)
            # Use Dixon-Coles as base if available, otherwise use Elo or H2H
            base_probs = results.get("poisson") or results.get("elo") or results.get("h2h")
            if base_probs is not None:
                results["dynamic"] = self._apply_dynamic_adjustment(
                    base_probs, adj_a, adj_b
                )
            else:
                # No base available, use neutral probabilities with adjustment
                neutral = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
                results["dynamic"] = self._apply_dynamic_adjustment(
                    neutral, adj_a, adj_b
                )
        except Exception:
            failed_models.append("dynamic")

        if not results:
            raise AllModelsFailedError(
                "All sub-models failed. Cannot produce a prediction."
            )

        # Redistribute weights excluding failed models
        effective_weights = self.weights
        for model_name in failed_models:
            effective_weights = effective_weights.redistribute_without(model_name)

        # Compute weighted combination with available models
        weight_map = {
            "poisson": effective_weights.poisson,
            "elo": effective_weights.elo,
            "h2h": effective_weights.h2h,
            "dynamic": effective_weights.dynamic,
        }

        win_a = 0.0
        draw = 0.0
        win_b = 0.0

        for model_name, probs in results.items():
            w = weight_map[model_name]
            win_a += w * probs[0]
            draw += w * probs[1]
            win_b += w * probs[2]

        # Normalize to ensure sum = 1.0
        total = win_a + draw + win_b
        if total > 0:
            win_a /= total
            draw /= total
            win_b /= total
        else:
            win_a = 1.0 / 3.0
            draw = 1.0 / 3.0
            win_b = 1.0 / 3.0

        return (win_a, draw, win_b)

    def update_weights(self, new_weights: EnsembleWeights) -> None:
        """Update weights after recalibration.

        Args:
            new_weights: New validated weights to use.

        Raises:
            ValueError: If new weights fail validation.
        """
        if not new_weights.validate():
            raise ValueError("New weights fail validation.")
        self.weights = new_weights

    def _matrix_to_probabilities(
        self, matrix: NDArray[np.float64]
    ) -> tuple[float, float, float]:
        """Convert a 5×5 score matrix to (win_a, draw, win_b) probabilities.

        win_a = sum of matrix[i][j] where i > j (team A scores more)
        draw = sum of matrix[i][j] where i == j (same score)
        win_b = sum of matrix[i][j] where i < j (team B scores more)

        Args:
            matrix: 5×5 probability matrix from Dixon-Coles model.

        Returns:
            Tuple of (win_a, draw, win_b) probabilities.
        """
        win_a = 0.0
        draw = 0.0
        win_b = 0.0

        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(cols):
                if i > j:
                    win_a += matrix[i, j]
                elif i == j:
                    draw += matrix[i, j]
                else:
                    win_b += matrix[i, j]

        # Normalize in case matrix doesn't sum to exactly 1.0
        total = win_a + draw + win_b
        if total > 0:
            win_a /= total
            draw /= total
            win_b /= total

        return (win_a, draw, win_b)

    def _apply_dynamic_adjustment(
        self,
        base_probs: tuple[float, float, float],
        adj_a: float,
        adj_b: float,
    ) -> tuple[float, float, float]:
        """Apply dynamic factor adjustments to base probabilities.

        The dynamic adjustment acts as a multiplier on win probabilities:
        - Positive adjustment for team A increases win_a probability
        - Positive adjustment for team B increases win_b probability

        Args:
            base_probs: Base (win_a, draw, win_b) probabilities.
            adj_a: Dynamic factor adjustment for team A.
            adj_b: Dynamic factor adjustment for team B.

        Returns:
            Adjusted and normalized (win_a, draw, win_b) probabilities.
        """
        win_a, draw, win_b = base_probs

        # Apply adjustments as multipliers
        win_a *= 1.0 + adj_a
        win_b *= 1.0 + adj_b

        # Ensure non-negative
        win_a = max(0.0, win_a)
        win_b = max(0.0, win_b)
        draw = max(0.0, draw)

        # Normalize
        total = win_a + draw + win_b
        if total > 0:
            win_a /= total
            draw /= total
            win_b /= total
        else:
            win_a = 1.0 / 3.0
            draw = 1.0 / 3.0
            win_b = 1.0 / 3.0

        return (win_a, draw, win_b)
