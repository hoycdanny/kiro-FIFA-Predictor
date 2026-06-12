"""
Property-based tests for the Ensemble Model.

**Validates: Requirements 7.4, 7.8**
"""

from hypothesis import given, settings, assume
from hypothesis.strategies import (
    composite,
    floats,
    sampled_from,
)

from src.engine.ensemble import EnsembleWeights
from src.utils.constants import WEIGHT_MIN, WEIGHT_MAX


MODEL_NAMES = ["poisson", "elo", "h2h", "dynamic"]


@composite
def valid_ensemble_weights(draw):
    """
    Strategy to generate 4 weights in [WEIGHT_MIN, WEIGHT_MAX] that sum to 1.0.

    Approach: generate 3 independent weights, derive the 4th to force sum = 1.0,
    then check all constraints.
    """
    w1 = draw(floats(min_value=WEIGHT_MIN, max_value=WEIGHT_MAX))
    w2 = draw(floats(min_value=WEIGHT_MIN, max_value=WEIGHT_MAX))
    w3 = draw(floats(min_value=WEIGHT_MIN, max_value=WEIGHT_MAX))
    w4 = 1.0 - w1 - w2 - w3

    # All weights must be in valid range
    assume(WEIGHT_MIN <= w4 <= WEIGHT_MAX)

    return EnsembleWeights(poisson=w1, elo=w2, h2h=w3, dynamic=w4)


class TestEnsembleWeightInvariant:
    """
    Property 8: Ensemble weight invariant.

    For any state of the EnsembleModel (initial, after recalibration, after
    model exclusion), each sub-model weight SHALL be in [0.10, 0.60] and the
    sum of all weights SHALL equal 1.00 (within ±1e-9).

    **Validates: Requirements 7.4, 7.8**
    """

    @given(weights=valid_ensemble_weights())
    @settings(max_examples=200)
    def test_valid_weights_pass_validation(self, weights: EnsembleWeights):
        """Any weights in [0.10, 0.60] summing to 1.0 must pass validate()."""
        assert weights.validate() is True, (
            f"Weights {weights} should pass validation but didn't. "
            f"Sum={weights.poisson + weights.elo + weights.h2h + weights.dynamic}"
        )

    @given(weights=valid_ensemble_weights())
    @settings(max_examples=200)
    def test_weight_bounds_invariant(self, weights: EnsembleWeights):
        """Each individual weight must be within [WEIGHT_MIN, WEIGHT_MAX]."""
        all_weights = [weights.poisson, weights.elo, weights.h2h, weights.dynamic]
        for w in all_weights:
            assert WEIGHT_MIN <= w <= WEIGHT_MAX, (
                f"Weight {w} out of bounds [{WEIGHT_MIN}, {WEIGHT_MAX}]"
            )

    @given(weights=valid_ensemble_weights())
    @settings(max_examples=200)
    def test_weight_sum_invariant(self, weights: EnsembleWeights):
        """Sum of all weights must equal 1.0 within ±1e-9."""
        total = weights.poisson + weights.elo + weights.h2h + weights.dynamic
        assert abs(total - 1.0) < 1e-9, (
            f"Weight sum {total} deviates from 1.0 beyond tolerance"
        )

    def test_default_weights_pass_validation(self):
        """Default EnsembleWeights (0.40, 0.25, 0.15, 0.20) must validate."""
        weights = EnsembleWeights()
        assert weights.validate() is True


class TestModelExclusionWeightRedistribution:
    """
    Property 13: Model exclusion weight redistribution.

    For any single excluded sub-model, the remaining models' redistributed
    weights SHALL maintain their original proportions relative to each other
    and sum to 1.00.

    **Validates: Requirements 7.4, 7.8**
    """

    @given(
        weights=valid_ensemble_weights(),
        excluded=sampled_from(MODEL_NAMES),
    )
    @settings(max_examples=200)
    def test_redistributed_weights_sum_to_one(
        self, weights: EnsembleWeights, excluded: str
    ):
        """After excluding one model, remaining weights must sum to 1.0."""
        result = weights.redistribute_without(excluded)

        all_weights = [result.poisson, result.elo, result.h2h, result.dynamic]
        total = sum(all_weights)
        assert abs(total - 1.0) < 1e-9, (
            f"Redistributed weights sum to {total}, expected 1.0. "
            f"Excluded: {excluded}, weights: {all_weights}"
        )

    @given(
        weights=valid_ensemble_weights(),
        excluded=sampled_from(MODEL_NAMES),
    )
    @settings(max_examples=200)
    def test_excluded_model_weight_is_zero(
        self, weights: EnsembleWeights, excluded: str
    ):
        """The excluded model's weight must be 0.0 after redistribution."""
        result = weights.redistribute_without(excluded)

        excluded_weight = getattr(result, excluded)
        assert excluded_weight == 0.0, (
            f"Excluded model '{excluded}' has weight {excluded_weight}, expected 0.0"
        )

    @given(
        weights=valid_ensemble_weights(),
        excluded=sampled_from(MODEL_NAMES),
    )
    @settings(max_examples=200)
    def test_proportions_maintained(self, weights: EnsembleWeights, excluded: str):
        """
        Remaining models' weights must maintain their original proportions
        relative to each other.
        """
        result = weights.redistribute_without(excluded)

        # Get the remaining model names (not excluded)
        remaining = [m for m in MODEL_NAMES if m != excluded]

        # Get original weights for remaining models
        original_remaining = {m: getattr(weights, m) for m in remaining}

        # Get redistributed weights for remaining models
        redistributed_remaining = {m: getattr(result, m) for m in remaining}

        # Check proportions: for any two remaining models i, j:
        # redistributed[i] / redistributed[j] == original[i] / original[j]
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                m_i, m_j = remaining[i], remaining[j]
                orig_i = original_remaining[m_i]
                orig_j = original_remaining[m_j]
                red_i = redistributed_remaining[m_i]
                red_j = redistributed_remaining[m_j]

                # Skip if original weights are zero (shouldn't happen with valid weights)
                if orig_j == 0.0 or red_j == 0.0:
                    continue

                orig_ratio = orig_i / orig_j
                red_ratio = red_i / red_j

                assert abs(orig_ratio - red_ratio) < 1e-9, (
                    f"Proportion not maintained for {m_i}/{m_j}: "
                    f"original ratio={orig_ratio}, redistributed ratio={red_ratio}. "
                    f"Excluded: {excluded}"
                )
