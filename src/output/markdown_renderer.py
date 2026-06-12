"""
Markdown Renderer - Formats prediction output for Kiro chat interface.

Produces structured Markdown that renders correctly in the Kiro chat
interface, including flag emojis, tables, and section headers.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.7
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.data_manager import TeamProfile
    from src.engine.monte_carlo import ChampionPrediction
    from src.engine.prediction_engine import GroupPrediction, MatchPrediction
    from src.utils.accuracy_tracker import AccuracyReport


# ============================================================================
# FLAG EMOJI MAPPING
# ============================================================================

FLAG_EMOJI: dict[str, str] = {
    "Argentina": "🇦🇷",
    "Australia": "🇦🇺",
    "Belgium": "🇧🇪",
    "Bolivia": "🇧🇴",
    "Brazil": "🇧🇷",
    "Cameroon": "🇨🇲",
    "Canada": "🇨🇦",
    "Chile": "🇨🇱",
    "Colombia": "🇨🇴",
    "Costa Rica": "🇨🇷",
    "Croatia": "🇭🇷",
    "Denmark": "🇩🇰",
    "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Ghana": "🇬🇭",
    "Honduras": "🇭🇳",
    "Indonesia": "🇮🇩",
    "Iran": "🇮🇷",
    "Italy": "🇮🇹",
    "Ivory Coast": "🇨🇮",
    "Jamaica": "🇯🇲",
    "Japan": "🇯🇵",
    "Mexico": "🇲🇽",
    "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿",
    "Nigeria": "🇳🇬",
    "Norway": "🇳🇴",
    "Panama": "🇵🇦",
    "Paraguay": "🇵🇾",
    "Peru": "🇵🇪",
    "Poland": "🇵🇱",
    "Portugal": "🇵🇹",
    "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Senegal": "🇸🇳",
    "Serbia": "🇷🇸",
    "Slovenia": "🇸🇮",
    "South Korea": "🇰🇷",
    "Spain": "🇪🇸",
    "Switzerland": "🇨🇭",
    "Trinidad and Tobago": "🇹🇹",
    "Tunisia": "🇹🇳",
    "Türkiye": "🇹🇷",
    "United States": "🇺🇸",
    "Uruguay": "🇺🇾",
    "Venezuela": "🇻🇪",
    "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
}


def _get_flag(team_name: str) -> str:
    """Get the flag emoji for a team, defaulting to 🏳️ if not found."""
    return FLAG_EMOJI.get(team_name, "🏳️")


# ============================================================================
# MARKDOWN RENDERER
# ============================================================================


class MarkdownRenderer:
    """Renders prediction outputs as structured Markdown for Kiro chat."""

    def render_match_prediction(self, prediction: "MatchPrediction") -> str:
        """Render a single match prediction.

        Includes: flag emojis, top 3 scores, W/D/L, confidence, xG, over/under.
        """
        flag_a = _get_flag(prediction.team_a)
        flag_b = _get_flag(prediction.team_b)

        lines: list[str] = []

        # Header
        lines.append(
            f"## ⚽ {flag_a} {prediction.team_a} vs {prediction.team_b} {flag_b}"
        )
        lines.append("")

        # Top 3 most likely scores
        lines.append("### 📊 Most Likely Scores")
        lines.append("")
        lines.append("| # | Score | Probability |")
        lines.append("|---|-------|-------------|")
        for i, (score_a, score_b, prob) in enumerate(prediction.top_scores, 1):
            lines.append(f"| {i} | {score_a}-{score_b} | {prob * 100:.1f}% |")
        lines.append("")

        # W/D/L probabilities
        lines.append("### 🎯 Win/Draw/Lose")
        lines.append("")
        lines.append(
            f"| {flag_a} {prediction.team_a} Win | Draw | {flag_b} {prediction.team_b} Win |"
        )
        lines.append("|---|---|---|")
        lines.append(
            f"| {prediction.win_prob * 100:.1f}% | "
            f"{prediction.draw_prob * 100:.1f}% | "
            f"{prediction.lose_prob * 100:.1f}% |"
        )
        lines.append("")

        # Confidence index
        confidence = prediction.confidence_index
        bar_filled = confidence // 5
        bar_empty = 20 - bar_filled
        confidence_bar = "█" * bar_filled + "░" * bar_empty
        lines.append(f"### 🔒 Confidence: {confidence}%")
        lines.append(f"`{confidence_bar}`")
        lines.append("")

        # Expected goals (xG)
        lines.append("### ⚡ Expected Goals (xG)")
        lines.append("")
        lines.append(
            f"- {flag_a} {prediction.team_a}: **{prediction.expected_goals_a:.2f}**"
        )
        lines.append(
            f"- {flag_b} {prediction.team_b}: **{prediction.expected_goals_b:.2f}**"
        )
        lines.append("")

        # Over/Under 2.5
        lines.append("### 📈 Over/Under 2.5 Goals")
        lines.append("")
        lines.append(
            f"- Over 2.5: **{prediction.over_2_5 * 100:.1f}%**"
        )
        lines.append(
            f"- Under 2.5: **{prediction.under_2_5 * 100:.1f}%**"
        )

        return "\n".join(lines)

    def render_group_standings(self, group_prediction: "GroupPrediction") -> str:
        """Render group standings as a markdown table.

        Columns: Rank, Team, P (points), W, D, L, GF, GA, GD
        """
        lines: list[str] = []

        lines.append(f"## 🏆 Group {group_prediction.group_id} Standings")
        lines.append("")
        lines.append("| Rank | Team | P | W | D | L | GF | GA | GD |")
        lines.append("|------|------|---|---|---|---|----|----|-----|")

        for i, standing in enumerate(group_prediction.standings, 1):
            flag = _get_flag(standing.team)
            status = ""
            if standing.qualification_status:
                status = f" {standing.qualification_status}"
            lines.append(
                f"| {i} | {flag} {standing.team}{status} | "
                f"{standing.points} | {standing.wins} | {standing.draws} | "
                f"{standing.losses} | {standing.goals_for} | "
                f"{standing.goals_against} | {standing.goal_difference} |"
            )

        return "\n".join(lines)

    def render_champion_prediction(self, champion: "ChampionPrediction") -> str:
        """Render champion prediction with top 5 and bracket visualization."""
        lines: list[str] = []

        # Header
        lines.append("## 🏆 Champion Prediction")
        lines.append("")

        # Top 5 contenders
        lines.append("### 🥇 Top 5 Contenders")
        lines.append("")
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        for i, (team, prob) in enumerate(champion.top_5):
            flag = _get_flag(team)
            lines.append(f"{medals[i]} {flag} **{team}** — {prob * 100:.1f}%")
        lines.append("")

        # Confidence
        lines.append(f"### 🔒 Simulation Confidence: {champion.confidence_index}%")
        lines.append("")

        # Bracket tree visualization
        lines.append("### 🌳 Tournament Bracket")
        lines.append("")
        lines.append("```")

        # Show round advancement probabilities for top 5 teams
        round_labels = {
            "round_of_32": "R32",
            "round_of_16": "R16",
            "quarter_finals": "QF",
            "semi_finals": "SF",
            "final": "F",
            "champion": "🏆",
        }

        # Header row
        header = f"{'Team':<20}"
        for label in round_labels.values():
            header += f" {label:>6}"
        lines.append(header)
        lines.append("-" * len(header))

        # Show probabilities for top 5 teams
        for team, _ in champion.top_5:
            row = f"{team:<20}"
            team_rounds = champion.round_probabilities.get(team, {})
            for round_key in round_labels:
                prob = team_rounds.get(round_key, 0.0)
                row += f" {prob * 100:>5.1f}%"
            lines.append(row)

        lines.append("```")

        return "\n".join(lines)

    def render_team_profile(self, team: "TeamProfile") -> str:
        """Render team profile in categorized sections."""
        flag = _get_flag(team.name)
        lines: list[str] = []

        # Header
        lines.append(f"## {flag} {team.name} ({team.name_zh})")
        lines.append("")

        # Basic Info
        lines.append("### 📋 Basic Info")
        lines.append("")
        lines.append(f"- **Confederation:** {team.confederation}")
        lines.append(f"- **FIFA Ranking:** #{team.fifa_ranking}")
        lines.append(f"- **FIFA Points:** {team.fifa_points:.1f}")
        lines.append(f"- **Elo Rating:** {team.elo_rating}")
        lines.append(f"- **Group:** {team.group}")
        lines.append("")

        # Recent Form
        lines.append("### 📈 Recent Form (Last 10 Matches)")
        lines.append("")
        lines.append(f"- **Avg Goals Scored:** {team.recent_goals_avg:.2f}")
        lines.append(f"- **Avg Goals Conceded:** {team.recent_conceded_avg:.2f}")
        lines.append(f"- **Win Rate:** {team.recent_win_rate:.1f}%")
        lines.append(f"- **Draw Rate:** {team.recent_draw_rate:.1f}%")
        lines.append(f"- **Loss Rate:** {team.recent_loss_rate:.1f}%")
        lines.append(f"- **Neutral Venue Win Rate:** {team.neutral_win_rate:.1f}%")
        lines.append("")

        # World Cup History
        lines.append("### 🏆 World Cup History")
        lines.append("")
        lines.append(f"- **Best Result:** {team.best_wc_result}")
        lines.append(f"- **vs Top 20 Win Rate:** {team.vs_top20_win_rate:.1f}%")
        lines.append(f"- **First Match Win Rate:** {team.wc_first_match_win_rate:.1f}%")
        lines.append(
            f"- **Penalty Shootout Win Rate:** {team.penalty_shootout_win_rate:.1f}%"
        )
        lines.append("")

        # Advanced Stats
        lines.append("### 📊 Advanced Stats")
        lines.append("")
        lines.append(f"- **First Half Goals:** {team.first_half_goal_pct:.1f}%")
        lines.append(f"- **Second Half Goals:** {team.second_half_goal_pct:.1f}%")
        lines.append(f"- **Clean Sheet Rate:** {team.clean_sheet_rate:.1f}%")
        lines.append(f"- **Failed to Score Rate:** {team.failed_to_score_rate:.1f}%")
        if team.current_win_streak > 0:
            lines.append(f"- **Current Win Streak:** {team.current_win_streak}")
        if team.current_loss_streak > 0:
            lines.append(f"- **Current Loss Streak:** {team.current_loss_streak}")

        return "\n".join(lines)

    def render_accuracy_report(self, report: "AccuracyReport") -> str:
        """Render accuracy report with metrics and breakdowns."""
        lines: list[str] = []

        lines.append("## 📊 Prediction Accuracy Report")
        lines.append("")
        lines.append(f"**Total Matches Evaluated:** {report.total_matches}")
        lines.append("")

        # Core metrics
        lines.append("### 🎯 Core Metrics")
        lines.append("")
        lines.append(f"- **Exact Score Hit Rate:** {report.exact_score_rate:.1f}%")
        lines.append(f"- **Direction Hit Rate:** {report.direction_rate:.1f}%")
        lines.append(f"- **Avg Goal Error:** {report.avg_goal_error:.2f}")
        lines.append("")

        # By coach style
        if report.by_coach_style:
            lines.append("### 🎭 By Coach Style")
            lines.append("")
            lines.append("| Style | Matches | Exact Score | Direction | Avg Error |")
            lines.append("|-------|---------|-------------|-----------|-----------|")
            for style_name, style_acc in report.by_coach_style.items():
                lines.append(
                    f"| {style_name} | {style_acc.total} | "
                    f"{style_acc.exact_score_rate:.1f}% | "
                    f"{style_acc.direction_rate:.1f}% | "
                    f"{style_acc.avg_goal_error:.2f} |"
                )
            lines.append("")

        # Confidence calibration
        if report.confidence_calibration:
            lines.append("### 🔒 Confidence Calibration")
            lines.append("")
            lines.append("| Band | Direction Hit Rate |")
            lines.append("|------|-------------------|")
            for band, rate in report.confidence_calibration.items():
                lines.append(f"| {band} | {rate:.1f}% |")
            lines.append("")

        # Cross-confederation
        if report.cross_confederation:
            lines.append("### 🌍 Cross-Confederation Accuracy")
            lines.append("")
            lines.append("| Matchup | Direction Hit Rate |")
            lines.append("|---------|-------------------|")
            for matchup, rate in report.cross_confederation.items():
                lines.append(f"| {matchup} | {rate:.1f}% |")

        return "\n".join(lines)

    def render_footer(self, data_updated_at: str, model_version: str) -> str:
        """Render the standard footer with timestamp and model version."""
        return f"> 📊 data_updated_at: {data_updated_at} | model_version: {model_version}"
