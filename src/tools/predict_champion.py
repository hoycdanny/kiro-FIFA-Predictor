"""
Predict champion MCP tool handler.

Thin wrapper that runs Monte Carlo simulation for the full knockout
bracket and formats the champion prediction output.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.data.data_manager import DataManager
    from src.engine.monte_carlo import MonteCarloSimulator
    from src.engine.prediction_engine import PredictionEngine
    from src.output.formatter import OutputFormatter


async def handle_predict_champion(
    simulations: int,
    engine: Optional["PredictionEngine"],
    monte_carlo: Optional["MonteCarloSimulator"],
    formatter: Optional["OutputFormatter"],
    data_manager: Optional["DataManager"],
) -> str:
    """Handle the predict_champion MCP tool invocation.

    Args:
        simulations: Number of Monte Carlo simulations to run.
        engine: Prediction engine instance.
        monte_carlo: Monte Carlo simulator instance.
        formatter: Output formatter instance.
        data_manager: Data manager instance.

    Returns:
        Formatted champion prediction with bracket or error message.
    """
    if engine is None or monte_carlo is None or formatter is None or data_manager is None:
        return "Error: Server not initialized. Please restart the server."

    # Validate simulation count
    if simulations < 100:
        simulations = 100
    elif simulations > 100000:
        simulations = 100000

    # Update simulator's simulation count
    monte_carlo.n_simulations = simulations

    # Determine qualified teams for knockout stage
    # Use group predictions to derive the 32 teams advancing
    try:
        qualified_teams = _get_qualified_teams(engine, data_manager)
    except Exception as e:
        return f"Error determining qualified teams: {e}"

    # Run Monte Carlo simulation
    try:
        champion_prediction = monte_carlo.simulate_tournament(
            qualified_teams=qualified_teams,
        )
    except ValueError as e:
        return f"Error: {e}"

    # Format output
    return formatter.format_champion_prediction(champion_prediction)


def _get_qualified_teams(
    engine: "PredictionEngine",
    data_manager: "DataManager",
) -> list[str]:
    """Determine the 32 teams that qualify for the knockout stage.

    In the 2026 World Cup format:
    - Top 2 from each group advance (24 teams)
    - Best 8 third-place teams advance (8 teams)
    - Total: 32 teams

    Uses group predictions if actual results are not yet available.

    Returns:
        List of 32 team names for the knockout bracket.
    """
    groups = data_manager.load_groups()
    qualified: list[str] = []

    # Predict each group and take top 2 + collect 3rd place teams
    third_place_teams: list[tuple[str, int, int, int]] = []  # (name, points, gd, gf)

    for group_id in sorted(groups.keys()):
        try:
            group_pred = engine.predict_group(group_id)
            standings = group_pred.standings

            # Top 2 qualify directly
            if len(standings) >= 2:
                qualified.append(standings[0].team)
                qualified.append(standings[1].team)

            # Track 3rd place for best-third selection
            if len(standings) >= 3:
                third = standings[2]
                third_place_teams.append(
                    (third.team, third.points, third.goal_difference, third.goals_for)
                )
        except Exception:
            # Fallback: take first 2 teams from group definition
            team_list = groups[group_id]
            qualified.extend(team_list[:2])
            if len(team_list) >= 3:
                third_place_teams.append((team_list[2], 0, 0, 0))

    # Select best 8 third-place teams
    third_place_teams.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    for team_name, _, _, _ in third_place_teams[:8]:
        qualified.append(team_name)

    # Ensure we have exactly 32
    if len(qualified) < 32:
        # Fill remaining spots from groups (4th place teams, ordered by group)
        for group_id in sorted(groups.keys()):
            if len(qualified) >= 32:
                break
            team_list = groups[group_id]
            for team in team_list:
                if team not in qualified and len(qualified) < 32:
                    qualified.append(team)

    return qualified[:32]
