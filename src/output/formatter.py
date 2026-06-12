"""
Output Formatter - Dispatches to appropriate renderer based on output environment.

Provides a unified interface for formatting prediction results, dispatching
to MarkdownRenderer (Kiro chat) or RichRenderer (CLI terminal) as needed.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.7
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.data.data_manager import TeamProfile
    from src.engine.monte_carlo import ChampionPrediction
    from src.engine.prediction_engine import GroupPrediction, MatchPrediction
    from src.utils.accuracy_tracker import AccuracyReport


MODEL_VERSION = "1.0.0"


class Renderer(Protocol):
    """Protocol for output renderers."""

    def render_match_prediction(self, prediction: "MatchPrediction") -> str: ...
    def render_group_standings(self, group_prediction: "GroupPrediction") -> str: ...
    def render_champion_prediction(self, champion: "ChampionPrediction") -> str: ...
    def render_team_profile(self, team: "TeamProfile") -> str: ...
    def render_accuracy_report(self, report: "AccuracyReport") -> str: ...
    def render_footer(self, data_updated_at: str, model_version: str) -> str: ...


class OutputFormatter:
    """Dispatches to appropriate renderer based on output environment."""

    def __init__(self, renderer: "Renderer | None" = None):
        if renderer is None:
            from src.output.markdown_renderer import MarkdownRenderer

            renderer = MarkdownRenderer()
        self.renderer = renderer

    def format_match_prediction(
        self, prediction: "MatchPrediction", data_updated_at: str | None = None
    ) -> str:
        """Format a single match prediction result with footer."""
        body = self.renderer.render_match_prediction(prediction)
        footer = self._make_footer(data_updated_at)
        return f"{body}\n\n{footer}"

    def format_group_standings(
        self, group_prediction: "GroupPrediction", data_updated_at: str | None = None
    ) -> str:
        """Format group standings as a table with footer."""
        body = self.renderer.render_group_standings(group_prediction)
        footer = self._make_footer(data_updated_at)
        return f"{body}\n\n{footer}"

    def format_champion_prediction(
        self, champion: "ChampionPrediction", data_updated_at: str | None = None
    ) -> str:
        """Format champion prediction with bracket tree and footer."""
        body = self.renderer.render_champion_prediction(champion)
        footer = self._make_footer(data_updated_at)
        return f"{body}\n\n{footer}"

    def format_team_profile(
        self, team: "TeamProfile", data_updated_at: str | None = None
    ) -> str:
        """Format a team profile in categorized sections with footer."""
        body = self.renderer.render_team_profile(team)
        footer = self._make_footer(data_updated_at)
        return f"{body}\n\n{footer}"

    def format_accuracy_report(
        self, report: "AccuracyReport", data_updated_at: str | None = None
    ) -> str:
        """Format accuracy report with footer."""
        body = self.renderer.render_accuracy_report(report)
        footer = self._make_footer(data_updated_at)
        return f"{body}\n\n{footer}"

    def _make_footer(self, data_updated_at: str | None = None) -> str:
        """Generate the standard footer with timestamp and model version."""
        if data_updated_at is None:
            data_updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.renderer.render_footer(data_updated_at, MODEL_VERSION)
