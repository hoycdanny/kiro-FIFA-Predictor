"""
Unit tests for the DataManager class.

Tests cover:
- Loading teams from actual data file
- Loading groups from actual data file
- Loading schedule from actual data file
- Startup validation (48 teams, 12 groups x 4 teams, required fields)
- Atomic write pattern for save_match_result and append_prediction_log
- Error handling for invalid/missing data
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.data.data_manager import (
    DataManager,
    DataValidationError,
    MatchResult,
    PredictionLogEntry,
    TeamProfile,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def real_data_dir() -> Path:
    """Return the real data directory path."""
    return Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def real_manager(real_data_dir: Path) -> DataManager:
    """DataManager pointing at the real data/ directory."""
    return DataManager(real_data_dir)


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with valid test files."""
    # Create minimal valid teams.json (48 teams)
    teams = []
    for i in range(48):
        teams.append({
            "name": f"Team{i}",
            "name_zh": f"隊伍{i}",
            "aliases": [f"T{i}"],
            "confederation": "UEFA",
            "fifa_ranking": i + 1,
            "fifa_points": 1500.0 + i,
            "elo_rating": 1800 + i,
            "group": chr(ord("A") + i // 4),
            "recent_goals_avg": 1.5,
            "recent_conceded_avg": 1.0,
            "recent_win_rate": 50.0,
            "recent_draw_rate": 25.0,
            "recent_loss_rate": 25.0,
            "neutral_win_rate": 45.0,
            "best_wc_result": "Group stage",
            "vs_top20_win_rate": 30.0,
            "wc_first_match_win_rate": 33.0,
            "penalty_shootout_win_rate": 50.0,
            "first_half_goal_pct": 45.0,
            "second_half_goal_pct": 55.0,
            "clean_sheet_rate": 25.0,
            "failed_to_score_rate": 20.0,
            "current_win_streak": 0,
            "current_loss_streak": 0,
            "last_match_date": None,
            "eliminated_by_2022": None,
        })

    with open(tmp_path / "teams.json", "w", encoding="utf-8") as f:
        json.dump(teams, f, ensure_ascii=False)

    # Create groups.json (12 groups, 4 teams each)
    groups = {}
    for i in range(12):
        group_id = chr(ord("A") + i)
        groups[group_id] = [f"Team{i * 4 + j}" for j in range(4)]

    with open(tmp_path / "groups.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False)

    # Create schedule.json
    schedule = {"matches": [
        {
            "match_id": "GS-A-1",
            "date": "2026-06-11",
            "team_a": "Team0",
            "team_b": "Team1",
            "stage": "group",
            "group": "A",
            "venue_city": "TestCity",
            "venue_country": "TestCountry",
        }
    ]}
    with open(tmp_path / "schedule.json", "w", encoding="utf-8") as f:
        json.dump(schedule, f, ensure_ascii=False)

    # Create match_results.json
    with open(tmp_path / "match_results.json", "w", encoding="utf-8") as f:
        json.dump({"matches": []}, f)

    # Create predictions_log.json
    with open(tmp_path / "predictions_log.json", "w", encoding="utf-8") as f:
        json.dump({"predictions": []}, f)

    return tmp_path


@pytest.fixture
def tmp_manager(tmp_data_dir: Path) -> DataManager:
    """DataManager pointing at temporary test data."""
    return DataManager(tmp_data_dir)


# ============================================================================
# TESTS: Loading real data
# ============================================================================


class TestLoadRealData:
    """Test DataManager with the actual project data files."""

    def test_load_teams_returns_48(self, real_manager: DataManager):
        teams = real_manager.load_teams()
        assert len(teams) == 48

    def test_load_teams_all_have_required_fields(self, real_manager: DataManager):
        teams = real_manager.load_teams()
        for team in teams:
            for field_name in DataManager.REQUIRED_TEAM_FIELDS:
                value = getattr(team, field_name)
                assert value is not None, f"{team.name} missing {field_name}"
                if isinstance(value, str):
                    assert value.strip(), f"{team.name} has empty {field_name}"

    def test_load_groups_returns_12_groups(self, real_manager: DataManager):
        groups = real_manager.load_groups()
        assert len(groups) == 12
        assert set(groups.keys()) == set("ABCDEFGHIJKL")

    def test_load_groups_each_has_4_teams(self, real_manager: DataManager):
        groups = real_manager.load_groups()
        for group_id, team_list in groups.items():
            assert len(team_list) == 4, f"Group {group_id} has {len(team_list)} teams"

    def test_load_schedule_returns_entries(self, real_manager: DataManager):
        schedule = real_manager.load_schedule()
        assert len(schedule) > 0
        assert schedule[0].match_id
        assert schedule[0].date

    def test_startup_validation_passes(self, real_manager: DataManager):
        """The actual data should pass startup validation."""
        real_manager.validate_startup()  # Should not raise


# ============================================================================
# TESTS: Startup validation
# ============================================================================


class TestStartupValidation:
    """Test startup validation logic."""

    def test_valid_data_passes(self, tmp_manager: DataManager):
        tmp_manager.validate_startup()  # Should not raise

    def test_wrong_team_count_fails(self, tmp_data_dir: Path):
        # Write only 10 teams
        teams = []
        for i in range(10):
            teams.append({
                "name": f"Team{i}",
                "name_zh": f"隊伍{i}",
                "aliases": [],
                "confederation": "UEFA",
                "fifa_ranking": i + 1,
                "fifa_points": 1500.0,
                "elo_rating": 1800,
                "group": "A",
                "recent_goals_avg": 1.5,
                "recent_conceded_avg": 1.0,
                "recent_win_rate": 50.0,
                "recent_draw_rate": 25.0,
                "recent_loss_rate": 25.0,
                "neutral_win_rate": 45.0,
                "best_wc_result": "Group stage",
                "vs_top20_win_rate": 30.0,
                "wc_first_match_win_rate": 33.0,
                "penalty_shootout_win_rate": 50.0,
                "first_half_goal_pct": 45.0,
                "second_half_goal_pct": 55.0,
                "clean_sheet_rate": 25.0,
                "failed_to_score_rate": 20.0,
            })
        with open(tmp_data_dir / "teams.json", "w", encoding="utf-8") as f:
            json.dump(teams, f)

        dm = DataManager(tmp_data_dir)
        with pytest.raises(DataValidationError, match="Expected 48 teams, got 10"):
            dm.validate_startup()

    def test_missing_group_fails(self, tmp_data_dir: Path):
        # Remove group L
        groups = {}
        for i in range(11):  # Only A-K
            group_id = chr(ord("A") + i)
            groups[group_id] = [f"Team{i * 4 + j}" for j in range(4)]

        with open(tmp_data_dir / "groups.json", "w", encoding="utf-8") as f:
            json.dump(groups, f)

        dm = DataManager(tmp_data_dir)
        with pytest.raises(DataValidationError, match="Missing groups"):
            dm.validate_startup()

    def test_wrong_group_size_fails(self, tmp_data_dir: Path):
        # Make group A have only 3 teams
        with open(tmp_data_dir / "groups.json", "r", encoding="utf-8") as f:
            groups = json.load(f)
        groups["A"] = ["Team0", "Team1", "Team2"]

        with open(tmp_data_dir / "groups.json", "w", encoding="utf-8") as f:
            json.dump(groups, f)

        dm = DataManager(tmp_data_dir)
        with pytest.raises(DataValidationError, match="Group A: expected 4 teams"):
            dm.validate_startup()

    def test_missing_required_field_fails(self, tmp_data_dir: Path):
        with open(tmp_data_dir / "teams.json", "r", encoding="utf-8") as f:
            teams = json.load(f)
        teams[0]["name_zh"] = ""  # Empty required field

        with open(tmp_data_dir / "teams.json", "w", encoding="utf-8") as f:
            json.dump(teams, f)

        dm = DataManager(tmp_data_dir)
        with pytest.raises(DataValidationError, match="missing required field 'name_zh'"):
            dm.validate_startup()

    def test_missing_teams_file_fails(self, tmp_data_dir: Path):
        os.remove(tmp_data_dir / "teams.json")
        dm = DataManager(tmp_data_dir)
        with pytest.raises(DataValidationError, match="Failed to load teams.json"):
            dm.validate_startup()


# ============================================================================
# TESTS: Atomic write
# ============================================================================


class TestAtomicWrite:
    """Test atomic write operations."""

    def test_save_match_result_persists(self, tmp_manager: DataManager, tmp_data_dir: Path):
        result = MatchResult(
            match_id="GS-A-1",
            date="2026-06-11",
            team_a="Team0",
            team_b="Team1",
            score_a=2,
            score_b=1,
            stage="group",
            group="A",
            venue_country="TestCountry",
        )
        tmp_manager.save_match_result(result)

        # Verify file content
        with open(tmp_data_dir / "match_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["matches"]) == 1
        assert data["matches"][0]["match_id"] == "GS-A-1"
        assert data["matches"][0]["score_a"] == 2

    def test_save_match_result_appends(self, tmp_manager: DataManager, tmp_data_dir: Path):
        for i in range(3):
            result = MatchResult(
                match_id=f"GS-A-{i}",
                date="2026-06-11",
                team_a="Team0",
                team_b="Team1",
                score_a=i,
                score_b=0,
                stage="group",
                group="A",
            )
            tmp_manager.save_match_result(result)

        with open(tmp_data_dir / "match_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["matches"]) == 3

    def test_append_prediction_log_persists(self, tmp_manager: DataManager, tmp_data_dir: Path):
        log_entry = PredictionLogEntry(
            timestamp="2026-06-10T14:30:00Z",
            match_id="GS-A-1",
            team_a="Team0",
            team_b="Team1",
            predicted_score=(2, 1),
            win_draw_lose=(0.55, 0.25, 0.20),
            confidence_index=72,
            coach_style="分析師",
            model_weights={"poisson": 0.40, "elo": 0.25, "h2h": 0.15, "dynamic": 0.20},
        )
        tmp_manager.append_prediction_log(log_entry)

        with open(tmp_data_dir / "predictions_log.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["match_id"] == "GS-A-1"
        assert data["predictions"][0]["timestamp"] == "2026-06-10T14:30:00Z"

    def test_atomic_write_no_temp_files_remain(self, tmp_manager: DataManager, tmp_data_dir: Path):
        result = MatchResult(
            match_id="GS-A-1",
            date="2026-06-11",
            team_a="Team0",
            team_b="Team1",
            score_a=1,
            score_b=0,
            stage="group",
        )
        tmp_manager.save_match_result(result)

        # No .tmp files should remain
        tmp_files = list(tmp_data_dir.glob("*.tmp"))
        assert tmp_files == []

    def test_atomic_write_produces_valid_json(self, tmp_manager: DataManager, tmp_data_dir: Path):
        result = MatchResult(
            match_id="GS-A-1",
            date="2026-06-11",
            team_a="Team0",
            team_b="Team1",
            score_a=3,
            score_b=2,
            stage="group",
            group="A",
            venue_country="United States",
        )
        tmp_manager.save_match_result(result)

        # File must be valid JSON
        with open(tmp_data_dir / "match_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)  # Should not raise
        assert isinstance(data, dict)
        assert "matches" in data


# ============================================================================
# TESTS: Edge cases
# ============================================================================


class TestEdgeCases:
    """Test edge case handling."""

    def test_load_match_results_empty(self, tmp_manager: DataManager):
        results = tmp_manager.load_match_results()
        assert results == []

    def test_missing_match_results_returns_empty(self, tmp_data_dir: Path):
        os.remove(tmp_data_dir / "match_results.json")
        dm = DataManager(tmp_data_dir)
        results = dm.load_match_results()
        assert results == []

    def test_missing_predictions_log_returns_empty(self, tmp_data_dir: Path):
        os.remove(tmp_data_dir / "predictions_log.json")
        dm = DataManager(tmp_data_dir)
        # append_prediction_log should create the file
        log_entry = PredictionLogEntry(
            timestamp="2026-06-10T14:30:00Z",
            match_id="GS-A-1",
            team_a="Team0",
            team_b="Team1",
            predicted_score=(1, 0),
            win_draw_lose=(0.50, 0.30, 0.20),
            confidence_index=60,
            coach_style="分析師",
        )
        dm.append_prediction_log(log_entry)
        assert (tmp_data_dir / "predictions_log.json").exists()
