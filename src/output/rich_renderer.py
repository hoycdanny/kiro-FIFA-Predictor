"""
Rich CLI Renderer - Uses Rich library for formatted terminal output.

Renders prediction results with tables, borders, color coding, and
alignment for CLI terminal display.

Requirements: 10.6
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from src.data.data_manager import TeamProfile
    from src.engine.monte_carlo import ChampionPrediction
    from src.engine.prediction_engine import GroupPrediction, MatchPrediction
    from src.utils.accuracy_tracker import AccuracyReport


def _prob_color(prob: float) -> str:
    """Return color name based on probability value (0-1 scale)."""
    if prob >= 0.5:
        return "green"
    elif prob >= 0.25:
        return "yellow"
    return "red"


def _pct_color(pct: float) -> str:
    """Return color name based on percentage value (0-100 scale)."""
    if pct >= 50.0:
        return "green"
    elif pct >= 25.0:
        return "yellow"
    return "red"


class RichRenderer:
    """Rich CLI output renderer with tables, borders, and colors."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console(record=True)

    def render_match_prediction(self, prediction: "MatchPrediction") -> str:
        """Render a single match prediction with Rich formatting."""
        self.console = Console(record=True)

        # Header with team names
        title = Text()
        title.append(f" {prediction.team_a}", style="bold cyan")
        title.append(" vs ", style="dim")
        title.append(f"{prediction.team_b} ", style="bold cyan")

        # Main prediction info
        win_color = _prob_color(prediction.win_prob)
        draw_color = _prob_color(prediction.draw_prob)
        lose_color = _prob_color(prediction.lose_prob)

        lines = []
        lines.append("")
        lines.append(f"  [bold]Win/Draw/Lose:[/bold]")
        lines.append(
            f"    [{win_color}]{prediction.team_a}: {prediction.win_prob * 100:.1f}%[/{win_color}]"
            f"  [dim]|[/dim]  "
            f"[{draw_color}]Draw: {prediction.draw_prob * 100:.1f}%[/{draw_color}]"
            f"  [dim]|[/dim]  "
            f"[{lose_color}]{prediction.team_b}: {prediction.lose_prob * 100:.1f}%[/{lose_color}]"
        )
        lines.append("")

        # Top 3 predicted scores
        lines.append(f"  [bold]Top Predicted Scores:[/bold]")
        for i, (sa, sb, prob) in enumerate(prediction.top_scores, 1):
            color = _prob_color(prob)
            lines.append(f"    {i}. {sa}-{sb}  [{color}]({prob * 100:.1f}%)[/{color}]")
        lines.append("")

        # Expected goals
        lines.append(
            f"  [bold]Expected Goals (xG):[/bold]  "
            f"{prediction.team_a} {prediction.expected_goals_a:.2f}  [dim]|[/dim]  "
            f"{prediction.team_b} {prediction.expected_goals_b:.2f}"
        )
        lines.append("")

        # Over/Under 2.5
        over_color = _prob_color(prediction.over_2_5)
        under_color = _prob_color(prediction.under_2_5)
        lines.append(
            f"  [bold]Over/Under 2.5:[/bold]  "
            f"[{over_color}]Over: {prediction.over_2_5 * 100:.1f}%[/{over_color}]"
            f"  [dim]|[/dim]  "
            f"[{under_color}]Under: {prediction.under_2_5 * 100:.1f}%[/{under_color}]"
        )
        lines.append("")

        # Confidence
        conf = prediction.confidence_index
        conf_color = "green" if conf >= 67 else ("yellow" if conf >= 34 else "red")
        lines.append(
            f"  [bold]Confidence:[/bold]  [{conf_color}]{conf}%[/{conf_color}]"
        )
        lines.append("")

        # Coach style
        lines.append(f"  [bold]Coach Style:[/bold]  [italic]{prediction.coach_style}[/italic]")

        content = "\n".join(lines)
        panel = Panel(content, title=title, border_style="blue", padding=(0, 1))
        self.console.print(panel)

        return self.console.export_text()

    def render_group_standings(self, group_prediction: "GroupPrediction") -> str:
        """Render group standings as a Rich table."""
        self.console = Console(record=True)

        table = Table(
            title=f"Group {group_prediction.group_id} Standings",
            title_style="bold white",
            border_style="blue",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Rank", justify="center", width=4)
        table.add_column("Team", justify="left", min_width=16)
        table.add_column("P", justify="center", width=3)
        table.add_column("W", justify="center", width=3)
        table.add_column("D", justify="center", width=3)
        table.add_column("L", justify="center", width=3)
        table.add_column("GF", justify="center", width=4)
        table.add_column("GA", justify="center", width=4)
        table.add_column("GD", justify="center", width=4)
        table.add_column("Pts", justify="center", width=4, style="bold")
        table.add_column("Status", justify="left", min_width=12)

        for i, standing in enumerate(group_prediction.standings, 1):
            # Color code by qualification status
            if "確定晉級" in standing.qualification_status:
                row_style = "green"
            elif "可能晉級" in standing.qualification_status:
                row_style = "yellow"
            else:
                row_style = ""

            gd_str = f"+{standing.goal_difference}" if standing.goal_difference > 0 else str(standing.goal_difference)

            table.add_row(
                str(i),
                standing.team,
                str(standing.played),
                str(standing.wins),
                str(standing.draws),
                str(standing.losses),
                str(standing.goals_for),
                str(standing.goals_against),
                gd_str,
                str(standing.points),
                standing.qualification_status,
                style=row_style,
            )

        self.console.print(table)

        # Match predictions summary
        self.console.print("")
        self.console.print("[bold]Match Predictions:[/bold]", style="white")
        for mp in group_prediction.match_predictions:
            top_score = mp.top_scores[0] if mp.top_scores else (0, 0, 0.0)
            self.console.print(
                f"  {mp.team_a} vs {mp.team_b}  →  "
                f"[cyan]{top_score[0]}-{top_score[1]}[/cyan] "
                f"[dim]({top_score[2] * 100:.1f}%)[/dim]"
            )

        return self.console.export_text()

    def render_champion_prediction(self, champion: "ChampionPrediction") -> str:
        """Render champion prediction with bracket and top 5."""
        self.console = Console(record=True)

        # Champion header
        champ_text = Text()
        champ_text.append("🏆 ", style="bold yellow")
        champ_text.append(f"Predicted Champion: {champion.predicted_champion}", style="bold green")
        champ_text.append(f"  ({champion.champion_probability * 100:.1f}%)", style="dim")
        self.console.print(Panel(champ_text, border_style="yellow"))

        # Top 5 table
        top5_table = Table(
            title="Top 5 Championship Contenders",
            title_style="bold white",
            border_style="cyan",
            show_header=True,
            header_style="bold cyan",
        )
        top5_table.add_column("#", justify="center", width=3)
        top5_table.add_column("Team", justify="left", min_width=16)
        top5_table.add_column("Win Probability", justify="center", min_width=14)

        for i, (team, prob) in enumerate(champion.top_5, 1):
            color = _prob_color(prob)
            top5_table.add_row(
                str(i),
                team,
                f"[{color}]{prob * 100:.1f}%[/{color}]",
            )

        self.console.print(top5_table)

        # Round advancement probabilities
        if champion.round_probabilities:
            self.console.print("")
            rounds_table = Table(
                title="Round Advancement Probabilities",
                title_style="bold white",
                border_style="blue",
                show_header=True,
                header_style="bold cyan",
            )
            rounds_table.add_column("Team", justify="left", min_width=16)

            # Determine round columns from the data
            round_names = ["R16", "QF", "SF", "Final", "Champion"]
            for rnd in round_names:
                rounds_table.add_column(rnd, justify="center", width=8)

            # Show top 10 teams by champion probability
            sorted_teams = sorted(
                champion.round_probabilities.items(),
                key=lambda x: x[1].get("Champion", x[1].get("champion", 0.0)),
                reverse=True,
            )[:10]

            for team, probs in sorted_teams:
                row = [team]
                for rnd in round_names:
                    # Handle both capitalized and lowercase key formats
                    p = probs.get(rnd, probs.get(rnd.lower(), 0.0))
                    color = _pct_color(p * 100)
                    row.append(f"[{color}]{p * 100:.1f}%[/{color}]")
                rounds_table.add_row(*row)

            self.console.print(rounds_table)

        # Confidence
        conf = champion.confidence_index
        conf_color = "green" if conf >= 67 else ("yellow" if conf >= 34 else "red")
        self.console.print("")
        self.console.print(
            f"[bold]Confidence Index:[/bold]  [{conf_color}]{conf}%[/{conf_color}]"
        )

        return self.console.export_text()

    def render_team_profile(self, team: "TeamProfile") -> str:
        """Render team profile in categorized sections."""
        self.console = Console(record=True)

        # Title
        title = Text(f" {team.name} ({team.name_zh}) ", style="bold white")

        # Basic Info section
        basic_table = Table(
            title="Basic Info",
            title_style="bold",
            border_style="blue",
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        basic_table.add_column("Field", style="bold cyan", min_width=20)
        basic_table.add_column("Value", min_width=20)

        basic_table.add_row("FIFA Ranking", str(team.fifa_ranking))
        basic_table.add_row("FIFA Points", f"{team.fifa_points:.1f}")
        basic_table.add_row("Elo Rating", str(team.elo_rating))
        basic_table.add_row("Confederation", team.confederation)
        basic_table.add_row("Group", team.group)

        # Recent Form section
        form_table = Table(
            title="Recent Form (Last 10)",
            title_style="bold",
            border_style="green",
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        form_table.add_column("Field", style="bold cyan", min_width=20)
        form_table.add_column("Value", min_width=20)

        form_table.add_row("Avg Goals Scored", f"{team.recent_goals_avg:.2f}")
        form_table.add_row("Avg Goals Conceded", f"{team.recent_conceded_avg:.2f}")
        form_table.add_row("Win Rate", f"{team.recent_win_rate:.1f}%")
        form_table.add_row("Draw Rate", f"{team.recent_draw_rate:.1f}%")
        form_table.add_row("Loss Rate", f"{team.recent_loss_rate:.1f}%")
        form_table.add_row("Neutral Venue Win Rate", f"{team.neutral_win_rate:.1f}%")

        streak_info = ""
        if team.current_win_streak > 0:
            streak_info = f"W{team.current_win_streak}"
        elif team.current_loss_streak > 0:
            streak_info = f"L{team.current_loss_streak}"
        else:
            streak_info = "—"
        form_table.add_row("Current Streak", streak_info)

        # World Cup History section
        wc_table = Table(
            title="World Cup History",
            title_style="bold",
            border_style="yellow",
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        wc_table.add_column("Field", style="bold cyan", min_width=20)
        wc_table.add_column("Value", min_width=20)

        wc_table.add_row("Best Result", team.best_wc_result)
        wc_table.add_row("vs Top 20 Win Rate", f"{team.vs_top20_win_rate:.1f}%")
        wc_table.add_row("WC First Match Win Rate", f"{team.wc_first_match_win_rate:.1f}%")
        wc_table.add_row("Penalty Shootout Win Rate", f"{team.penalty_shootout_win_rate:.1f}%")

        # Advanced Stats section
        adv_table = Table(
            title="Advanced Stats",
            title_style="bold",
            border_style="magenta",
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        adv_table.add_column("Field", style="bold cyan", min_width=20)
        adv_table.add_column("Value", min_width=20)

        adv_table.add_row("1st Half Goal %", f"{team.first_half_goal_pct:.1f}%")
        adv_table.add_row("2nd Half Goal %", f"{team.second_half_goal_pct:.1f}%")
        adv_table.add_row("Clean Sheet Rate", f"{team.clean_sheet_rate:.1f}%")
        adv_table.add_row("Failed to Score Rate", f"{team.failed_to_score_rate:.1f}%")

        # Render all sections in a panel
        panel = Panel(
            "",
            title=title,
            border_style="cyan",
        )
        self.console.print(panel)
        self.console.print(basic_table)
        self.console.print("")
        self.console.print(form_table)
        self.console.print("")
        self.console.print(wc_table)
        self.console.print("")
        self.console.print(adv_table)

        return self.console.export_text()

    def render_accuracy_report(self, report: "AccuracyReport") -> str:
        """Render accuracy report with tables and color coding."""
        self.console = Console(record=True)

        # Overview panel
        overview_lines = []
        overview_lines.append(f"  [bold]Total Matches Evaluated:[/bold]  {report.total_matches}")
        overview_lines.append("")

        esr_color = _pct_color(report.exact_score_rate)
        overview_lines.append(
            f"  [bold]Exact Score Rate:[/bold]  [{esr_color}]{report.exact_score_rate:.1f}%[/{esr_color}]"
        )

        dr_color = _pct_color(report.direction_rate)
        overview_lines.append(
            f"  [bold]Direction Rate (W/D/L):[/bold]  [{dr_color}]{report.direction_rate:.1f}%[/{dr_color}]"
        )

        overview_lines.append(
            f"  [bold]Avg Goal Error:[/bold]  {report.avg_goal_error:.2f}"
        )

        content = "\n".join(overview_lines)
        self.console.print(Panel(content, title="Accuracy Report", border_style="blue"))

        # By coach style
        if report.by_coach_style:
            style_table = Table(
                title="Accuracy by Coach Style",
                title_style="bold",
                border_style="green",
                show_header=True,
                header_style="bold cyan",
            )
            style_table.add_column("Style", justify="left", min_width=12)
            style_table.add_column("Matches", justify="center", width=8)
            style_table.add_column("Exact Score", justify="center", width=12)
            style_table.add_column("Direction", justify="center", width=10)
            style_table.add_column("Avg Error", justify="center", width=10)

            for style_name, acc in report.by_coach_style.items():
                esr_c = _pct_color(acc.exact_score_rate)
                dr_c = _pct_color(acc.direction_rate)
                style_table.add_row(
                    style_name,
                    str(acc.total),
                    f"[{esr_c}]{acc.exact_score_rate:.1f}%[/{esr_c}]",
                    f"[{dr_c}]{acc.direction_rate:.1f}%[/{dr_c}]",
                    f"{acc.avg_goal_error:.2f}",
                )

            self.console.print("")
            self.console.print(style_table)

        # Confidence calibration
        if report.confidence_calibration:
            cal_table = Table(
                title="Confidence Calibration",
                title_style="bold",
                border_style="yellow",
                show_header=True,
                header_style="bold cyan",
            )
            cal_table.add_column("Band", justify="center", min_width=12)
            cal_table.add_column("Direction Rate", justify="center", min_width=14)

            for band, rate in report.confidence_calibration.items():
                color = _pct_color(rate)
                cal_table.add_row(band, f"[{color}]{rate:.1f}%[/{color}]")

            self.console.print("")
            self.console.print(cal_table)

        # Cross-confederation
        if report.cross_confederation:
            conf_table = Table(
                title="Cross-Confederation Accuracy",
                title_style="bold",
                border_style="magenta",
                show_header=True,
                header_style="bold cyan",
            )
            conf_table.add_column("Matchup", justify="left", min_width=20)
            conf_table.add_column("Direction Rate", justify="center", min_width=14)

            for matchup, rate in report.cross_confederation.items():
                color = _pct_color(rate)
                conf_table.add_row(matchup, f"[{color}]{rate:.1f}%[/{color}]")

            self.console.print("")
            self.console.print(conf_table)

        return self.console.export_text()

    def render_footer(self, data_updated_at: str, model_version: str) -> str:
        """Render the standard footer with timestamp and model version."""
        self.console = Console(record=True)
        self.console.print(
            f"[dim]───────────────────────────────────────[/dim]"
        )
        self.console.print(
            f"[dim]Data updated: {data_updated_at}  |  Model: v{model_version}[/dim]"
        )
        return self.console.export_text()
