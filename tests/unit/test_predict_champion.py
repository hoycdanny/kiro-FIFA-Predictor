"""
Tests for the predict_champion MCP tool handler.

Verifies:
- Correct error when dependencies are None
- Simulation count is clamped within valid bounds
- Qualified teams are determined from group predictions
- Monte Carlo simulation is invoked with correct parameters
- Output is formatted via OutputFormatter
- Handles fallback when group predictions fail
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.tools.predict_champion import handle_predict_champion, _get_qualified_teams


@dataclass
class MockGroupStanding:
    """Mock group standing for testing."""
    team: str
    points: int
    goal_difference: int
    goals_for: int


@dataclass
class MockGroupPrediction:
    """Mock group prediction for testing."""
    group_id: str
    standings: list


class TestHandlePredictChampion:
    """Test the handle_predict_champion function."""

    @pytest.mark.asyncio
    async def test_returns_error_when_engine_is_none(self):
        """Should return error message when engine is None."""
        result = await handle_predict_champion(
            simulations=10000,
            engine=None,
            monte_carlo=MagicMock(),
            formatter=MagicMock(),
            data_manager=MagicMock(),
        )
        assert "Error" in result
        assert "not initialized" in result

    @pytest.mark.asyncio
    async def test_returns_error_when_monte_carlo_is_none(self):
        """Should return error message when monte_carlo is None."""
        result = await handle_predict_champion(
            simulations=10000,
            engine=MagicMock(),
            monte_carlo=None,
            formatter=MagicMock(),
            data_manager=MagicMock(),
        )
        assert "Error" in result
        assert "not initialized" in result

    @pytest.mark.asyncio
    async def test_returns_error_when_formatter_is_none(self):
        """Should return error message when formatter is None."""
        result = await handle_predict_champion(
            simulations=10000,
            engine=MagicMock(),
            monte_carlo=MagicMock(),
            formatter=None,
            data_manager=MagicMock(),
        )
        assert "Error" in result
        assert "not initialized" in result

    @pytest.mark.asyncio
    async def test_returns_error_when_data_manager_is_none(self):
        """Should return error message when data_manager is None."""
        result = await handle_predict_champion(
            simulations=10000,
            engine=MagicMock(),
            monte_carlo=MagicMock(),
            formatter=MagicMock(),
            data_manager=None,
        )
        assert "Error" in result
        assert "not initialized" in result

    @pytest.mark.asyncio
    async def test_clamps_simulations_minimum(self):
        """Should clamp simulations to minimum 100."""
        mock_mc = MagicMock()
        mock_engine = MagicMock()
        mock_formatter = MagicMock()
        mock_dm = MagicMock()

        # Set up mock groups (12 groups with 4 teams each)
        groups = {chr(65 + i): [f"Team{i}_{j}" for j in range(4)] for i in range(12)}
        mock_dm.load_groups.return_value = groups

        # Set up mock group predictions
        def mock_predict_group(group_id):
            teams = groups[group_id]
            standings = [
                MockGroupStanding(team=teams[0], points=9, goal_difference=5, goals_for=8),
                MockGroupStanding(team=teams[1], points=6, goal_difference=2, goals_for=5),
                MockGroupStanding(team=teams[2], points=3, goal_difference=0, goals_for=3),
                MockGroupStanding(team=teams[3], points=0, goal_difference=-7, goals_for=1),
            ]
            return MockGroupPrediction(group_id=group_id, standings=standings)

        mock_engine.predict_group.side_effect = mock_predict_group

        mock_result = MagicMock()
        mock_mc.simulate_tournament.return_value = mock_result
        mock_formatter.format_champion_prediction.return_value = "Champion: TeamA"

        await handle_predict_champion(
            simulations=10,  # Below minimum
            engine=mock_engine,
            monte_carlo=mock_mc,
            formatter=mock_formatter,
            data_manager=mock_dm,
        )

        # Should have been clamped to 100
        assert mock_mc.n_simulations == 100

    @pytest.mark.asyncio
    async def test_clamps_simulations_maximum(self):
        """Should clamp simulations to maximum 100000."""
        mock_mc = MagicMock()
        mock_engine = MagicMock()
        mock_formatter = MagicMock()
        mock_dm = MagicMock()

        groups = {chr(65 + i): [f"Team{i}_{j}" for j in range(4)] for i in range(12)}
        mock_dm.load_groups.return_value = groups

        def mock_predict_group(group_id):
            teams = groups[group_id]
            standings = [
                MockGroupStanding(team=teams[0], points=9, goal_difference=5, goals_for=8),
                MockGroupStanding(team=teams[1], points=6, goal_difference=2, goals_for=5),
                MockGroupStanding(team=teams[2], points=3, goal_difference=0, goals_for=3),
                MockGroupStanding(team=teams[3], points=0, goal_difference=-7, goals_for=1),
            ]
            return MockGroupPrediction(group_id=group_id, standings=standings)

        mock_engine.predict_group.side_effect = mock_predict_group
        mock_mc.simulate_tournament.return_value = MagicMock()
        mock_formatter.format_champion_prediction.return_value = "Champion: TeamA"

        await handle_predict_champion(
            simulations=500000,  # Above maximum
            engine=mock_engine,
            monte_carlo=mock_mc,
            formatter=mock_formatter,
            data_manager=mock_dm,
        )

        assert mock_mc.n_simulations == 100000

    @pytest.mark.asyncio
    async def test_successful_prediction_flow(self):
        """Should execute full flow: groups -> qualified -> simulate -> format."""
        mock_mc = MagicMock()
        mock_engine = MagicMock()
        mock_formatter = MagicMock()
        mock_dm = MagicMock()

        groups = {chr(65 + i): [f"Team{i}_{j}" for j in range(4)] for i in range(12)}
        mock_dm.load_groups.return_value = groups

        def mock_predict_group(group_id):
            teams = groups[group_id]
            standings = [
                MockGroupStanding(team=teams[0], points=9, goal_difference=5, goals_for=8),
                MockGroupStanding(team=teams[1], points=6, goal_difference=2, goals_for=5),
                MockGroupStanding(team=teams[2], points=3, goal_difference=0, goals_for=3),
                MockGroupStanding(team=teams[3], points=0, goal_difference=-7, goals_for=1),
            ]
            return MockGroupPrediction(group_id=group_id, standings=standings)

        mock_engine.predict_group.side_effect = mock_predict_group
        mock_champion_result = MagicMock()
        mock_mc.simulate_tournament.return_value = mock_champion_result
        mock_formatter.format_champion_prediction.return_value = "🏆 Predicted Champion: Brazil (22.5%)"

        result = await handle_predict_champion(
            simulations=10000,
            engine=mock_engine,
            monte_carlo=mock_mc,
            formatter=mock_formatter,
            data_manager=mock_dm,
        )

        # Verify simulation count was set
        assert mock_mc.n_simulations == 10000

        # Verify simulate_tournament was called with 32 teams
        mock_mc.simulate_tournament.assert_called_once()
        call_args = mock_mc.simulate_tournament.call_args
        qualified = call_args[1]["qualified_teams"]
        assert len(qualified) == 32

        # Verify formatter was called with simulation result
        mock_formatter.format_champion_prediction.assert_called_once_with(mock_champion_result)

        # Verify output
        assert "Predicted Champion" in result

    @pytest.mark.asyncio
    async def test_handles_simulate_tournament_value_error(self):
        """Should return error message when simulate_tournament raises ValueError."""
        mock_mc = MagicMock()
        mock_engine = MagicMock()
        mock_formatter = MagicMock()
        mock_dm = MagicMock()

        groups = {chr(65 + i): [f"Team{i}_{j}" for j in range(4)] for i in range(12)}
        mock_dm.load_groups.return_value = groups

        def mock_predict_group(group_id):
            teams = groups[group_id]
            standings = [
                MockGroupStanding(team=teams[0], points=9, goal_difference=5, goals_for=8),
                MockGroupStanding(team=teams[1], points=6, goal_difference=2, goals_for=5),
                MockGroupStanding(team=teams[2], points=3, goal_difference=0, goals_for=3),
                MockGroupStanding(team=teams[3], points=0, goal_difference=-7, goals_for=1),
            ]
            return MockGroupPrediction(group_id=group_id, standings=standings)

        mock_engine.predict_group.side_effect = mock_predict_group
        mock_mc.simulate_tournament.side_effect = ValueError("Expected 32 qualified teams, got 0")

        result = await handle_predict_champion(
            simulations=10000,
            engine=mock_engine,
            monte_carlo=mock_mc,
            formatter=mock_formatter,
            data_manager=mock_dm,
        )

        assert "Error" in result


class TestGetQualifiedTeams:
    """Test the _get_qualified_teams helper function."""

    def test_returns_32_teams(self):
        """Should return exactly 32 qualified teams."""
        mock_engine = MagicMock()
        mock_dm = MagicMock()

        groups = {chr(65 + i): [f"Team{i}_{j}" for j in range(4)] for i in range(12)}
        mock_dm.load_groups.return_value = groups

        def mock_predict_group(group_id):
            teams = groups[group_id]
            standings = [
                MockGroupStanding(team=teams[0], points=9, goal_difference=5, goals_for=8),
                MockGroupStanding(team=teams[1], points=6, goal_difference=2, goals_for=5),
                MockGroupStanding(team=teams[2], points=3, goal_difference=0, goals_for=3),
                MockGroupStanding(team=teams[3], points=0, goal_difference=-7, goals_for=1),
            ]
            return MockGroupPrediction(group_id=group_id, standings=standings)

        mock_engine.predict_group.side_effect = mock_predict_group

        qualified = _get_qualified_teams(mock_engine, mock_dm)

        assert len(qualified) == 32

    def test_top_2_from_each_group_qualify(self):
        """Top 2 from each of 12 groups should be in qualified teams."""
        mock_engine = MagicMock()
        mock_dm = MagicMock()

        groups = {chr(65 + i): [f"Team{i}_{j}" for j in range(4)] for i in range(12)}
        mock_dm.load_groups.return_value = groups

        def mock_predict_group(group_id):
            teams = groups[group_id]
            standings = [
                MockGroupStanding(team=teams[0], points=9, goal_difference=5, goals_for=8),
                MockGroupStanding(team=teams[1], points=6, goal_difference=2, goals_for=5),
                MockGroupStanding(team=teams[2], points=3, goal_difference=0, goals_for=3),
                MockGroupStanding(team=teams[3], points=0, goal_difference=-7, goals_for=1),
            ]
            return MockGroupPrediction(group_id=group_id, standings=standings)

        mock_engine.predict_group.side_effect = mock_predict_group

        qualified = _get_qualified_teams(mock_engine, mock_dm)

        # All top-2 teams from each group should be in qualified
        for i in range(12):
            group_id = chr(65 + i)
            teams = groups[group_id]
            assert teams[0] in qualified, f"1st place team from group {group_id} not qualified"
            assert teams[1] in qualified, f"2nd place team from group {group_id} not qualified"

    def test_best_8_third_place_teams_qualify(self):
        """Best 8 third-place teams should qualify based on points, GD, GF."""
        mock_engine = MagicMock()
        mock_dm = MagicMock()

        groups = {chr(65 + i): [f"Team{i}_{j}" for j in range(4)] for i in range(12)}
        mock_dm.load_groups.return_value = groups

        def mock_predict_group(group_id):
            teams = groups[group_id]
            idx = ord(group_id) - 65
            # Give third-place teams varying points for ranking
            third_points = 6 - (idx % 4)  # Varies from 3 to 6
            standings = [
                MockGroupStanding(team=teams[0], points=9, goal_difference=5, goals_for=8),
                MockGroupStanding(team=teams[1], points=7, goal_difference=2, goals_for=5),
                MockGroupStanding(team=teams[2], points=third_points, goal_difference=idx, goals_for=3 + idx),
                MockGroupStanding(team=teams[3], points=0, goal_difference=-7, goals_for=1),
            ]
            return MockGroupPrediction(group_id=group_id, standings=standings)

        mock_engine.predict_group.side_effect = mock_predict_group

        qualified = _get_qualified_teams(mock_engine, mock_dm)

        # Should have exactly 32 teams
        assert len(qualified) == 32

        # Count how many third-place teams are in qualified
        third_place_in_qualified = 0
        for i in range(12):
            teams = groups[chr(65 + i)]
            if teams[2] in qualified:
                third_place_in_qualified += 1

        assert third_place_in_qualified == 8

    def test_fallback_when_predict_group_fails(self):
        """Should use fallback (first teams from group) when predict_group raises."""
        mock_engine = MagicMock()
        mock_dm = MagicMock()

        groups = {chr(65 + i): [f"Team{i}_{j}" for j in range(4)] for i in range(12)}
        mock_dm.load_groups.return_value = groups

        # All predictions fail
        mock_engine.predict_group.side_effect = Exception("Model not loaded")

        qualified = _get_qualified_teams(mock_engine, mock_dm)

        # Should still return 32 teams using fallback
        assert len(qualified) == 32

    def test_no_duplicate_teams(self):
        """Qualified teams should have no duplicates."""
        mock_engine = MagicMock()
        mock_dm = MagicMock()

        groups = {chr(65 + i): [f"Team{i}_{j}" for j in range(4)] for i in range(12)}
        mock_dm.load_groups.return_value = groups

        def mock_predict_group(group_id):
            teams = groups[group_id]
            standings = [
                MockGroupStanding(team=teams[0], points=9, goal_difference=5, goals_for=8),
                MockGroupStanding(team=teams[1], points=6, goal_difference=2, goals_for=5),
                MockGroupStanding(team=teams[2], points=3, goal_difference=0, goals_for=3),
                MockGroupStanding(team=teams[3], points=0, goal_difference=-7, goals_for=1),
            ]
            return MockGroupPrediction(group_id=group_id, standings=standings)

        mock_engine.predict_group.side_effect = mock_predict_group

        qualified = _get_qualified_teams(mock_engine, mock_dm)

        assert len(qualified) == len(set(qualified)), "Duplicate teams in qualified list"
