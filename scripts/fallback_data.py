"""Fallback data generator.

Generates a validated, frozen copy of teams.json with metadata for use
when the external data source is unavailable (Requirement 9.4).

Usage:
    python scripts/fallback_data.py
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_TEAM_COUNT = 48


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    sha256.update(path.read_bytes())
    return sha256.hexdigest()


def _validate_source(data: list | dict) -> tuple[list[dict], list[str]]:
    """Validate the source teams.json data.

    The source can be either a top-level list of teams or an object
    with a "teams" key containing the list.

    Returns a tuple of (teams_list, validation_errors).
    """
    errors: list[str] = []

    # Handle both formats: top-level list or {"teams": [...]}
    if isinstance(data, list):
        teams = data
    elif isinstance(data, dict) and "teams" in data:
        teams = data["teams"]
        if not isinstance(teams, list):
            errors.append(f"'teams' must be a list, got {type(teams).__name__}")
            return [], errors
    else:
        errors.append(
            "Source must be a JSON array or object with 'teams' key"
        )
        return [], errors

    if len(teams) != EXPECTED_TEAM_COUNT:
        errors.append(
            f"Expected {EXPECTED_TEAM_COUNT} teams, found {len(teams)}"
        )

    # Check each team has required fields
    required_fields = {"name", "fifa_ranking", "elo_rating", "confederation"}
    for i, team in enumerate(teams):
        if not isinstance(team, dict):
            errors.append(f"Team at index {i} is not a dict")
            continue
        missing = required_fields - set(team.keys())
        if missing:
            team_name = team.get("name", f"index {i}")
            errors.append(
                f"Team '{team_name}' missing fields: {sorted(missing)}"
            )

    return teams, errors


def generate_fallback() -> bool:
    """Generate a frozen copy of teams.json for offline use.

    Steps:
    1. Verify source file exists
    2. Load and validate source data (ensure 48 teams, required fields)
    3. Create canonical fallback with _metadata header
    4. Write to data/teams_fallback.json

    Returns:
        True if fallback was generated successfully, False otherwise.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    source = data_dir / "teams.json"

    # Step 1: Verify source exists
    if not source.exists():
        print(f"ERROR: Source file not found: {source}", file=sys.stderr)
        return False

    # Step 2: Load and validate
    try:
        raw_content = source.read_text(encoding="utf-8")
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in source file: {e}", file=sys.stderr)
        return False

    teams, errors = _validate_source(data)
    if errors:
        print("ERROR: Source data validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return False

    print(f"✓ Source validated: {len(teams)} teams found")

    # Step 3: Build fallback with metadata
    source_hash = _compute_file_hash(source)
    timestamp = datetime.now(timezone.utc).isoformat()

    fallback_data = {
        "_metadata": {
            "generated_at": timestamp,
            "source_file": "data/teams.json",
            "source_hash_sha256": source_hash,
            "team_count": len(teams),
            "version": "1.0.0",
        },
        "teams": teams,
    }

    # Step 4: Write canonical fallback
    canonical = data_dir / "teams_fallback.json"
    canonical.write_text(
        json.dumps(fallback_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✓ Fallback data generated: {canonical}")
    print(f"  Timestamp: {timestamp}")
    print(f"  Source hash: {source_hash[:16]}...")
    return True


if __name__ == "__main__":
    success = generate_fallback()
    sys.exit(0 if success else 1)
