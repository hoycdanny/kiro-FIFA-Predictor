"""
MCP Server entry point for the FIFA Predictor Power.

Initializes the FastMCP server, registers all 6 MCP tools, and wires up
the DataManager, PredictionEngine, and supporting components at startup.

Requirements: 9.1, 9.2, 9.3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from src.data.data_manager import DataManager, DataValidationError
from src.engine.ensemble import EnsembleModel, EnsembleWeights
from src.engine.monte_carlo import MonteCarloSimulator
from src.engine.prediction_engine import PredictionEngine
from src.output.formatter import OutputFormatter
from src.utils.accuracy_tracker import AccuracyTracker
from src.utils.team_matcher import TeamMatcher
from src.utils.validator import InputValidator


# ============================================================================
# MODULE-LEVEL GLOBALS (initialized at startup)
# ============================================================================

mcp = FastMCP("fifa-predictor")

# These are initialized during startup()
_data_manager: Optional[DataManager] = None
_prediction_engine: Optional[PredictionEngine] = None
_monte_carlo: Optional[MonteCarloSimulator] = None
_output_formatter: Optional[OutputFormatter] = None
_accuracy_tracker: Optional[AccuracyTracker] = None
_input_validator: Optional[InputValidator] = None


# ============================================================================
# STARTUP
# ============================================================================


def _get_data_dir() -> Path:
    """Resolve the data directory path relative to project root.

    The project root is the parent of the `src/` directory.
    """
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data"


async def startup() -> None:
    """Startup flow: load data, validate, initialize engine.

    Steps:
    1. Initialize DataManager with data/ path
    2. Run validate_startup() — abort if fails
    3. Initialize EnsembleModel with default weights
    4. Initialize PredictionEngine with DataManager and EnsembleModel
    5. Initialize MonteCarloSimulator
    6. Initialize OutputFormatter
    7. Initialize AccuracyTracker
    8. Initialize InputValidator with TeamMatcher

    Raises:
        SystemExit: If data validation fails at startup.
    """
    global _data_manager, _prediction_engine, _monte_carlo
    global _output_formatter, _accuracy_tracker, _input_validator

    data_dir = _get_data_dir()

    # Step 1: Initialize DataManager
    _data_manager = DataManager(data_dir)

    # Step 2: Validate data integrity — abort if fails
    try:
        _data_manager.validate_startup()
    except DataValidationError as e:
        print(f"FATAL: Startup validation failed:\n{e}", file=sys.stderr)
        sys.exit(1)

    # Step 3: Initialize EnsembleModel with default weights
    ensemble = EnsembleModel(weights=EnsembleWeights())

    # Step 4: Initialize PredictionEngine
    _prediction_engine = PredictionEngine(
        data_manager=_data_manager,
        ensemble=ensemble,
    )

    # Step 5: Initialize MonteCarloSimulator
    teams = _data_manager.load_teams()
    teams_dict = {t.name: t for t in teams}
    _monte_carlo = MonteCarloSimulator(
        ensemble=ensemble,
        teams=teams_dict,
    )

    # Step 6: Initialize OutputFormatter (defaults to Markdown renderer)
    _output_formatter = OutputFormatter()

    # Step 7: Initialize AccuracyTracker
    _accuracy_tracker = AccuracyTracker(data_manager=_data_manager)

    # Step 8: Initialize InputValidator with TeamMatcher
    team_names = [t.name for t in teams]
    # Include non-participant teams for friendly match support
    try:
        np_teams = _data_manager.load_non_participant_teams()
        np_team_names = [t.name for t in np_teams]
        all_known_names = team_names + np_team_names
    except (FileNotFoundError, Exception):
        all_known_names = team_names
    team_matcher = TeamMatcher(all_known_names)
    _input_validator = InputValidator(team_matcher=team_matcher)


def create_server() -> FastMCP:
    """Create and return the configured MCP server instance.

    This function returns the FastMCP server with all tools registered.
    Call startup() before running the server to initialize dependencies.

    Returns:
        The configured FastMCP server instance.
    """
    return mcp


# ============================================================================
# MCP TOOL REGISTRATIONS
# ============================================================================


@mcp.tool()
async def predict_match(
    team_a: str,
    team_b: str,
    coach_style: str | None = None,
) -> str:
    """Predict a single match between two teams.

    Provides win/draw/lose probabilities, top 3 most likely scores,
    confidence index, over/under 2.5 goals, and expected goals for each team.
    Supports three coach styles: analyst (分析師), contrarian (反向思考者),
    and tactician (戰術家).

    Args:
        team_a: Name of the first team (English or Chinese).
        team_b: Name of the second team (English or Chinese).
        coach_style: Optional coaching analysis style. One of: "分析師",
            "反向思考者", "戰術家", or keywords like "conservative",
            "aggressive", "balanced".

    Returns:
        Formatted prediction result as a string.
    """
    from src.tools.predict_match import handle_predict_match

    return await handle_predict_match(
        team_a=team_a,
        team_b=team_b,
        coach_style=coach_style,
        engine=_prediction_engine,
        validator=_input_validator,
        formatter=_output_formatter,
        data_manager=_data_manager,
    )


@mcp.tool()
async def predict_group(group_id: str) -> str:
    """Predict group stage standings and match results.

    Simulates all 6 round-robin matches for the specified group and
    produces a ranked standings table with points, goals, and
    qualification status.

    Args:
        group_id: Group identifier (A through L).

    Returns:
        Formatted group standings table as a string.
    """
    from src.tools.predict_group import handle_predict_group

    return await handle_predict_group(
        group_id=group_id,
        engine=_prediction_engine,
        validator=_input_validator,
        formatter=_output_formatter,
    )


@mcp.tool()
async def predict_champion(simulations: int = 10000) -> str:
    """Predict the World Cup champion using Monte Carlo simulation.

    Runs thousands of tournament simulations through the full knockout
    bracket (round of 32 through final) to determine championship
    probabilities for all qualified teams.

    Args:
        simulations: Number of Monte Carlo simulations to run (default 10000).

    Returns:
        Formatted champion prediction with bracket and top-5 probabilities.
    """
    from src.tools.predict_champion import handle_predict_champion

    return await handle_predict_champion(
        simulations=simulations,
        engine=_prediction_engine,
        monte_carlo=_monte_carlo,
        formatter=_output_formatter,
        data_manager=_data_manager,
    )


@mcp.tool()
async def update_results(
    match_id: str | None = None,
    manual_result: str | None = None,
) -> str:
    """Update match results and recalibrate prediction model.

    Fetches the latest match results from the data source (or accepts
    manual input), compares with predictions, adjusts model weights,
    and updates team dynamic factors.

    Args:
        match_id: Optional specific match ID to update.
        manual_result: Optional manual result string (format: "team_a score_a - score_b team_b").

    Returns:
        Recalibration report showing weight changes and accuracy metrics.
    """
    from src.tools.update_results import handle_update_results

    return await handle_update_results(
        match_id=match_id,
        manual_result=manual_result,
        data_manager=_data_manager,
        ensemble=_prediction_engine.ensemble if _prediction_engine else None,
        formatter=_output_formatter,
    )


@mcp.tool()
async def accuracy_stats() -> str:
    """Display prediction accuracy statistics.

    Shows exact score hit rate, win/draw/lose direction accuracy,
    average goal error, breakdowns by coach style, confidence
    calibration analysis, and cross-confederation accuracy.

    Returns:
        Formatted accuracy report or insufficient data message.
    """
    from src.tools.accuracy_stats import handle_accuracy_stats

    return await handle_accuracy_stats(
        accuracy_tracker=_accuracy_tracker,
        formatter=_output_formatter,
    )


@mcp.tool()
async def team_info(team_name: str) -> str:
    """Query detailed team profile information.

    Displays comprehensive team data including FIFA ranking, Elo rating,
    recent form, World Cup history, and advanced statistics.
    Supports fuzzy matching with English names, Chinese names, and
    abbreviations.

    Args:
        team_name: Team name to look up (supports fuzzy matching).

    Returns:
        Formatted team profile or error with suggestions.
    """
    from src.tools.team_info import handle_team_info

    return await handle_team_info(
        team_name=team_name,
        validator=_input_validator,
        formatter=_output_formatter,
        data_manager=_data_manager,
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point for the FIFA Predictor MCP server."""
    import asyncio

    asyncio.run(startup())
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
