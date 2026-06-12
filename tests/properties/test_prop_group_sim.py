"""
Property-based tests for group simulation and Monte Carlo tournament.

Tests Properties 6 and 7 from the design document:
- Property 6: Group simulation structural invariants
- Property 7: Tournament round probability consistency

**Validates: Requirements 2.1, 2.2, 2.5, 3.2**
"""

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from, integers

from src.data.data_manager import DataManager
from src.engine.ensemble import EnsembleModel
from src.engine.monte_carlo import MonteCarloSimulator
from src.engine.prediction_engine import PredictionEngine
from src.utils.constants import GROUP_ASSIGNMENTS


# Use real data directory
DATA_DIR = Path(__file__).parents[2] / "data"

# All valid group IDs
GROUP_IDS = list(GROUP_ASSIGNMENTS.keys())

# Load actual team names from the data file
_data_manager = DataManager(data_dir=DATA_DIR)
_teams = _data_manager.load_teams()
AVAILABLE_TEAMS = sorted(set(t.name for t in _teams))


@pytest.fixture(scope="module")
def engine():
    """Initialize PredictionEngine once for all tests in this module."""
    data_manager = DataManager(data_dir=DATA_DIR)
    ensemble = EnsembleModel()
    return PredictionEngine(data_manager=data_manager, ensemble=ensemble)


@pytest.fixture(scope="module")
def monte_carlo_deps():
    """Initialize Monte Carlo dependencies once for all tests."""
    data_manager = DataManager(data_dir=DATA_DIR)
    ensemble = EnsembleModel()
    teams = data_manager.load_teams()
    teams_dict = {t.name: t for t in teams}
    return ensemble, teams_dict


class TestGroupSimulationStructuralInvariants:
    """Property 6: Group simulation structural invariants.

    For any group prediction (groups A through L), the result SHALL satisfy:
    - Exactly 4 teams are ranked
    - Exactly 6 match predictions are produced
    - For each team: matches_played = 3, W + D + L = 3,
      points = 3×W + D, and goal_difference = goals_for − goals_against.

    **Validates: Requirements 2.1, 2.2, 2.5**
    """

    @given(group_id=sampled_from(GROUP_IDS))
    @settings(max_examples=12, deadline=None)
    def test_group_has_exactly_4_teams_ranked(self, engine, group_id):
        """Group prediction must have exactly 4 teams in standings."""
        result = engine.predict_group(group_id)

        assert len(result.standings) == 4, (
            f"Group {group_id}: expected 4 teams in standings, got {len(result.standings)}"
        )

    @given(group_id=sampled_from(GROUP_IDS))
    @settings(max_examples=12, deadline=None)
    def test_group_has_exactly_6_match_predictions(self, engine, group_id):
        """Group prediction must have exactly 6 match predictions (round-robin of 4)."""
        result = engine.predict_group(group_id)

        assert len(result.match_predictions) == 6, (
            f"Group {group_id}: expected 6 match predictions, got {len(result.match_predictions)}"
        )

    @given(group_id=sampled_from(GROUP_IDS))
    @settings(max_examples=12, deadline=None)
    def test_each_team_plays_exactly_3_matches(self, engine, group_id):
        """Each team in a group must have played = 3."""
        result = engine.predict_group(group_id)

        for standing in result.standings:
            assert standing.played == 3, (
                f"Group {group_id}, team {standing.team}: "
                f"played = {standing.played}, expected 3"
            )

    @given(group_id=sampled_from(GROUP_IDS))
    @settings(max_examples=12, deadline=None)
    def test_wdl_sum_equals_3_for_each_team(self, engine, group_id):
        """For each team: W + D + L = 3 (number of matches played)."""
        result = engine.predict_group(group_id)

        for standing in result.standings:
            wdl_sum = standing.wins + standing.draws + standing.losses
            assert wdl_sum == 3, (
                f"Group {group_id}, team {standing.team}: "
                f"W+D+L = {wdl_sum} (W={standing.wins}, D={standing.draws}, L={standing.losses}), "
                f"expected 3"
            )

    @given(group_id=sampled_from(GROUP_IDS))
    @settings(max_examples=12, deadline=None)
    def test_points_formula_correct(self, engine, group_id):
        """For each team: points = 3×W + D."""
        result = engine.predict_group(group_id)

        for standing in result.standings:
            expected_points = 3 * standing.wins + standing.draws
            assert standing.points == expected_points, (
                f"Group {group_id}, team {standing.team}: "
                f"points = {standing.points}, expected 3×{standing.wins} + {standing.draws} = {expected_points}"
            )

    @given(group_id=sampled_from(GROUP_IDS))
    @settings(max_examples=12, deadline=None)
    def test_goal_difference_formula_correct(self, engine, group_id):
        """For each team: goal_difference = goals_for − goals_against."""
        result = engine.predict_group(group_id)

        for standing in result.standings:
            expected_gd = standing.goals_for - standing.goals_against
            assert standing.goal_difference == expected_gd, (
                f"Group {group_id}, team {standing.team}: "
                f"GD = {standing.goal_difference}, "
                f"expected GF({standing.goals_for}) - GA({standing.goals_against}) = {expected_gd}"
            )


class TestTournamentRoundProbabilityConsistency:
    """Property 7: Tournament round probability consistency.

    For any Monte Carlo tournament simulation result with 32 qualified teams,
    the sum of all teams' probabilities of reaching a given round SHALL equal
    the number of available slots in that round (16 for round-of-16, 8 for
    quarter-finals, 4 for semi-finals, 2 for final, 1 for champion),
    within ±1% tolerance.

    **Validates: Requirements 3.2**
    """

    # Expected number of teams advancing to each round
    ROUND_SLOTS = {
        "round_of_32": 16,   # 16 winners from 16 matches
        "round_of_16": 8,    # 8 winners from 8 matches
        "quarter_finals": 4, # 4 winners from 4 matches
        "semi_finals": 2,    # 2 winners from 2 matches
        "final": 1,          # 1 winner from 1 match
        "champion": 1,       # 1 champion
    }

    @given(
        n_simulations=sampled_from([100, 200, 500]),
    )
    @settings(max_examples=3, deadline=None)
    def test_round_probability_sums_equal_slots(self, monte_carlo_deps, n_simulations):
        """Sum of all teams' probabilities for a round = number of slots in that round."""
        ensemble, teams_dict = monte_carlo_deps

        # Use first 32 teams from AVAILABLE_TEAMS
        qualified_teams = AVAILABLE_TEAMS[:32]

        simulator = MonteCarloSimulator(
            ensemble=ensemble,
            teams=teams_dict,
            n_simulations=n_simulations,
        )

        result = simulator.simulate_tournament(qualified_teams)

        # Check each round
        for round_name, expected_slots in self.ROUND_SLOTS.items():
            total_prob = sum(
                result.round_probabilities[team][round_name]
                for team in qualified_teams
            )

            # Tolerance: ±1% of expected slots (converted to probability scale)
            # e.g., 16 slots → tolerance of 0.16
            tolerance = expected_slots * 0.01
            # Use absolute tolerance of at least 0.1 for low slot counts
            tolerance = max(tolerance, 0.1)

            assert abs(total_prob - expected_slots) <= tolerance, (
                f"Round '{round_name}': sum of probabilities = {total_prob:.4f}, "
                f"expected {expected_slots} ± {tolerance:.4f} "
                f"(n_simulations={n_simulations})"
            )

    @given(
        n_simulations=sampled_from([100, 200, 500]),
    )
    @settings(max_examples=3, deadline=None)
    def test_confidence_index_valid_for_tournament(self, monte_carlo_deps, n_simulations):
        """Tournament confidence_index must be an integer in [0, 100]."""
        ensemble, teams_dict = monte_carlo_deps

        qualified_teams = AVAILABLE_TEAMS[:32]

        simulator = MonteCarloSimulator(
            ensemble=ensemble,
            teams=teams_dict,
            n_simulations=n_simulations,
        )

        result = simulator.simulate_tournament(qualified_teams)

        assert isinstance(result.confidence_index, int), (
            f"confidence_index is {type(result.confidence_index)}, expected int"
        )
        assert 0 <= result.confidence_index <= 100, (
            f"confidence_index = {result.confidence_index}, expected [0, 100]"
        )
