"""
Data manager module for the FIFA Predictor Power.

Contains core data models (TeamProfile, MatchResult, ScheduleEntry,
PredictionLogEntry, PredictionError) and the DataManager class for
loading, saving, and validating data.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================================
# EXCEPTIONS
# ============================================================================


class DataValidationError(Exception):
    """Raised when startup data validation fails."""
    pass


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class TeamProfile:
    """Complete profile for a participating team."""

    # Basic info
    name: str                          # English canonical name
    name_zh: str                       # Chinese name (Traditional)
    aliases: list[str]                 # Other aliases/abbreviations
    confederation: str                 # UEFA/CONMEBOL/CONCACAF/CAF/AFC/OFC
    fifa_ranking: int                  # FIFA ranking
    fifa_points: float                 # FIFA points
    elo_rating: int                    # Elo rating
    group: str                         # Group (A-L)

    # Recent form (last 10 matches)
    recent_goals_avg: float            # Average goals scored (2 decimals)
    recent_conceded_avg: float         # Average goals conceded (2 decimals)
    recent_win_rate: float             # Win rate (percentage)
    recent_draw_rate: float            # Draw rate (percentage)
    recent_loss_rate: float            # Loss rate (percentage)

    # Neutral venue
    neutral_win_rate: float            # Neutral venue win rate (percentage)

    # World Cup history
    best_wc_result: str                # Best WC result
    vs_top20_win_rate: float           # Win rate vs top 20 (percentage)
    wc_first_match_win_rate: float     # WC first match win rate (percentage)
    penalty_shootout_win_rate: float   # Penalty shootout win rate (percentage)

    # Advanced stats
    first_half_goal_pct: float         # First half goal percentage
    second_half_goal_pct: float        # Second half goal percentage
    clean_sheet_rate: float            # Clean sheet rate (percentage)
    failed_to_score_rate: float        # Failed to score rate (percentage)

    # Dynamic data (updated during tournament)
    current_win_streak: int = 0
    current_loss_streak: int = 0
    last_match_date: Optional[str] = None    # ISO 8601
    eliminated_by_2022: Optional[str] = None  # 2022 eliminator team name


@dataclass
class MatchResult:
    """A recorded match result."""

    match_id: str
    date: str                    # ISO 8601
    team_a: str
    team_b: str
    score_a: int
    score_b: int
    stage: str                   # "group" or "knockout"
    group: Optional[str] = None
    venue_country: str = ""


@dataclass
class ScheduleEntry:
    """A scheduled match entry."""

    match_id: str
    date: str                    # ISO 8601
    team_a: str
    team_b: str
    stage: str                   # "group" or "knockout"
    group: Optional[str] = None
    venue_city: str = ""
    venue_country: str = ""


@dataclass
class PredictionLogEntry:
    """A logged prediction record."""

    timestamp: str                     # ISO 8601
    match_id: str
    team_a: str
    team_b: str
    predicted_score: tuple[int, int]
    win_draw_lose: tuple[float, float, float]
    confidence_index: int
    coach_style: str
    model_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionError:
    """Structured error for prediction failures."""

    error_code: str              # e.g., "INVALID_TEAM", "INVALID_GROUP", "INVALID_STYLE"
    message: str                 # User-friendly message
    suggestions: list[str] = field(default_factory=list)


# ============================================================================
# DATA MANAGER
# ============================================================================


class DataManager:
    """Manages loading, saving, and validating data files."""

    REQUIRED_TEAM_FIELDS = [
        "name", "name_zh", "aliases", "confederation",
        "fifa_ranking", "fifa_points", "elo_rating", "group",
        "recent_goals_avg", "recent_conceded_avg",
        "recent_win_rate", "recent_draw_rate", "recent_loss_rate",
        "neutral_win_rate", "best_wc_result", "vs_top20_win_rate",
        "wc_first_match_win_rate", "penalty_shootout_win_rate",
        "first_half_goal_pct", "second_half_goal_pct",
        "clean_sheet_rate", "failed_to_score_rate",
    ]

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load_teams(self) -> list[TeamProfile]:
        """Load teams.json and deserialize into TeamProfile list."""
        filepath = self.data_dir / "teams.json"
        data = self._read_json(filepath)

        # Handle both formats: plain array or {"teams": [...]}
        if isinstance(data, list):
            entries = data
        else:
            entries = data.get("teams", [])

        teams: list[TeamProfile] = []
        for entry in entries:
            team = self._entry_to_team_profile(entry)
            teams.append(team)

        return teams

    def load_non_participant_teams(self) -> list[TeamProfile]:
        """Load non_participant_teams.json for friendly match predictions.

        Returns an empty list if the file does not exist.
        """
        filepath = self.data_dir / "non_participant_teams.json"
        if not filepath.exists():
            return []

        data = self._read_json(filepath)

        if isinstance(data, list):
            entries = data
        else:
            entries = data.get("teams", [])

        teams: list[TeamProfile] = []
        for entry in entries:
            team = self._entry_to_team_profile(entry)
            teams.append(team)

        return teams

    def _entry_to_team_profile(self, entry: dict) -> TeamProfile:
        """Convert a JSON entry dict into a TeamProfile dataclass."""
        return TeamProfile(
            name=entry["name"],
            name_zh=entry["name_zh"],
            aliases=entry.get("aliases", []),
            confederation=entry["confederation"],
            fifa_ranking=entry["fifa_ranking"],
            fifa_points=entry["fifa_points"],
            elo_rating=entry["elo_rating"],
            group=entry["group"],
            recent_goals_avg=entry["recent_goals_avg"],
            recent_conceded_avg=entry["recent_conceded_avg"],
            recent_win_rate=entry["recent_win_rate"],
            recent_draw_rate=entry["recent_draw_rate"],
            recent_loss_rate=entry["recent_loss_rate"],
            neutral_win_rate=entry["neutral_win_rate"],
            best_wc_result=entry["best_wc_result"],
            vs_top20_win_rate=entry["vs_top20_win_rate"],
            wc_first_match_win_rate=entry["wc_first_match_win_rate"],
            penalty_shootout_win_rate=entry["penalty_shootout_win_rate"],
            first_half_goal_pct=entry["first_half_goal_pct"],
            second_half_goal_pct=entry["second_half_goal_pct"],
            clean_sheet_rate=entry["clean_sheet_rate"],
            failed_to_score_rate=entry["failed_to_score_rate"],
            current_win_streak=entry.get("current_win_streak", 0),
            current_loss_streak=entry.get("current_loss_streak", 0),
            last_match_date=entry.get("last_match_date"),
            eliminated_by_2022=entry.get("eliminated_by_2022"),
        )

    def load_groups(self) -> dict[str, list[str]]:
        """Load groups.json, return {group_id: [team_names]}."""
        filepath = self.data_dir / "groups.json"
        return self._read_json(filepath)

    def load_schedule(self) -> list[ScheduleEntry]:
        """Load schedule.json."""
        filepath = self.data_dir / "schedule.json"
        data = self._read_json(filepath)

        entries: list[ScheduleEntry] = []
        for item in data.get("matches", []):
            entry = ScheduleEntry(
                match_id=item["match_id"],
                date=item["date"],
                team_a=item["team_a"],
                team_b=item["team_b"],
                stage=item["stage"],
                group=item.get("group"),
                venue_city=item.get("venue_city", ""),
                venue_country=item.get("venue_country", ""),
            )
            entries.append(entry)

        return entries

    def load_match_results(self) -> list[MatchResult]:
        """Load match_results.json."""
        filepath = self.data_dir / "match_results.json"
        data = self._read_json(filepath)

        results: list[MatchResult] = []
        for item in data.get("matches", []):
            result = MatchResult(
                match_id=item["match_id"],
                date=item["date"],
                team_a=item["team_a"],
                team_b=item["team_b"],
                score_a=item["score_a"],
                score_b=item["score_b"],
                stage=item["stage"],
                group=item.get("group"),
                venue_country=item.get("venue_country", ""),
            )
            results.append(result)

        return results

    def save_match_result(self, result: MatchResult) -> None:
        """
        Save a match result using atomic write.
        Appends result to match_results.json.
        """
        filepath = self.data_dir / "match_results.json"
        data = self._read_json(filepath)

        matches = data.get("matches", [])
        matches.append({
            "match_id": result.match_id,
            "date": result.date,
            "team_a": result.team_a,
            "team_b": result.team_b,
            "score_a": result.score_a,
            "score_b": result.score_b,
            "stage": result.stage,
            "group": result.group,
            "venue_country": result.venue_country,
        })
        data["matches"] = matches

        self._atomic_write(filepath, data)

    def append_prediction_log(self, log_entry: PredictionLogEntry) -> None:
        """Append a prediction record to predictions_log.json."""
        filepath = self.data_dir / "predictions_log.json"
        data = self._read_json(filepath)

        predictions = data.get("predictions", [])
        predictions.append({
            "timestamp": log_entry.timestamp,
            "match_id": log_entry.match_id,
            "team_a": log_entry.team_a,
            "team_b": log_entry.team_b,
            "predicted_score": list(log_entry.predicted_score),
            "win_draw_lose": list(log_entry.win_draw_lose),
            "confidence_index": log_entry.confidence_index,
            "coach_style": log_entry.coach_style,
            "model_weights": log_entry.model_weights,
        })
        data["predictions"] = predictions

        self._atomic_write(filepath, data)

    def validate_teams(self, teams: list[TeamProfile]) -> list[str]:
        """
        Validate team data integrity.
        Returns list of error messages (empty if valid).
        """
        errors: list[str] = []

        if len(teams) < 32:
            errors.append(f"Expected at least 32 teams, got {len(teams)}")

        for team in teams:
            for field_name in self.REQUIRED_TEAM_FIELDS:
                value = getattr(team, field_name, None)
                if value is None or (isinstance(value, str) and not value.strip()):
                    errors.append(
                        f"Team '{team.name}': missing required field '{field_name}'"
                    )

        return errors

    def validate_groups(self, groups: dict[str, list[str]]) -> list[str]:
        """
        Validate group data integrity.
        Returns list of error messages (empty if valid).
        """
        errors: list[str] = []
        expected_groups = set("ABCDEFGHIJKL")

        if set(groups.keys()) != expected_groups:
            missing = expected_groups - set(groups.keys())
            extra = set(groups.keys()) - expected_groups
            if missing:
                errors.append(f"Missing groups: {sorted(missing)}")
            if extra:
                errors.append(f"Unexpected groups: {sorted(extra)}")

        for group_id, team_list in groups.items():
            if len(team_list) != 4:
                errors.append(
                    f"Group {group_id}: expected 4 teams, got {len(team_list)}"
                )

        return errors

    def validate_startup(self) -> None:
        """
        Perform startup validation.
        Verifies:
        - Exactly 48 teams are loaded
        - 12 groups with exactly 4 teams each
        - No missing required fields in TeamProfile

        Raises DataValidationError with specific error details if validation fails.
        """
        errors: list[str] = []

        # Load and validate teams
        try:
            teams = self.load_teams()
            team_errors = self.validate_teams(teams)
            errors.extend(team_errors)
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            errors.append(f"Failed to load teams.json: {e}")

        # Load and validate groups
        try:
            groups = self.load_groups()
            group_errors = self.validate_groups(groups)
            errors.extend(group_errors)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            errors.append(f"Failed to load groups.json: {e}")

        # Load schedule (verify it can be parsed)
        try:
            self.load_schedule()
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            errors.append(f"Failed to load schedule.json: {e}")

        if errors:
            raise DataValidationError(
                f"Startup validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def _read_json(self, filepath: Path) -> dict | list:
        """Read and parse a JSON file. Returns empty dict for missing writable files."""
        if not filepath.exists():
            # For writable files (results, predictions log), return empty structure
            if filepath.name in ("match_results.json", "predictions_log.json"):
                return {}
            raise FileNotFoundError(f"Required data file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _atomic_write(self, filepath: Path, data: dict) -> None:
        """Atomic write: write to tempfile then rename to target."""
        fd, tmp_path = tempfile.mkstemp(
            dir=str(filepath.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(filepath))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
