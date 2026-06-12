"""
Property-based tests for data layer integrity.

Tests Properties 17 and 18 from the design document:
- Property 17: Atomic write data integrity
- Property 18: Prediction log entry completeness

Validates: Requirements 9.5, 9.6
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.data_manager import DataManager, PredictionLogEntry


# ============================================================================
# Strategies
# ============================================================================

# Strategy for JSON-serializable primitive values
json_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(min_size=0, max_size=50),
)

# Recursive strategy for JSON-serializable data objects (dicts)
json_values = st.recursive(
    json_primitives,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=20), children, max_size=5),
    ),
    max_leaves=20,
)

json_objects = st.dictionaries(
    st.text(min_size=1, max_size=20),
    json_values,
    min_size=1,
    max_size=10,
)

# Strategy for valid ISO 8601 timestamps
iso_timestamps = st.datetimes(
    min_value=datetime(2024, 1, 1),
    max_value=datetime(2030, 12, 31),
    timezones=st.just(timezone.utc),
).map(lambda dt: dt.isoformat())

# Strategy for non-empty match IDs
match_ids = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=20,
)

# Strategy for team names
team_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=30,
)

# Strategy for coach style strings
coach_styles = st.sampled_from(["分析師", "反向思考者", "戰術家"])

# Strategy for model weights dicts
model_weights = st.fixed_dictionaries({
    "poisson": st.floats(min_value=0.10, max_value=0.60),
    "elo": st.floats(min_value=0.10, max_value=0.60),
    "h2h": st.floats(min_value=0.10, max_value=0.60),
    "dynamic": st.floats(min_value=0.10, max_value=0.60),
})

# Strategy for PredictionLogEntry instances
prediction_log_entries = st.builds(
    PredictionLogEntry,
    timestamp=iso_timestamps,
    match_id=match_ids,
    team_a=team_names,
    team_b=team_names,
    predicted_score=st.tuples(
        st.integers(min_value=0, max_value=10),
        st.integers(min_value=0, max_value=10),
    ),
    win_draw_lose=st.tuples(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    ),
    confidence_index=st.integers(min_value=0, max_value=100),
    coach_style=coach_styles,
    model_weights=model_weights,
)


# ============================================================================
# Property 17: Atomic write data integrity
# ============================================================================


class TestAtomicWriteDataIntegrity:
    """
    Property 17: Atomic write data integrity

    For any valid data object written via the atomic write mechanism,
    the target file SHALL contain valid JSON that deserializes to an
    object equal to the written object.

    **Validates: Requirements 9.5**
    """

    @given(data=json_objects)
    @settings(max_examples=100)
    def test_atomic_write_produces_valid_json_equal_to_input(
        self, data: dict
    ) -> None:
        """
        For any valid JSON-serializable dict, writing via _atomic_write
        and reading back should yield an identical object.

        **Validates: Requirements 9.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dm = DataManager(data_dir=tmp_path)
            target_file = tmp_path / "test_output.json"

            # Write using the atomic write mechanism
            dm._atomic_write(target_file, data)

            # Read back and verify
            assert target_file.exists(), "Target file should exist after atomic write"

            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Must be valid JSON
            deserialized = json.loads(content)

            # Deserialized object must equal the written object
            assert deserialized == data


# ============================================================================
# Property 18: Prediction log entry completeness
# ============================================================================


class TestPredictionLogEntryCompleteness:
    """
    Property 18: Prediction log entry completeness

    For any prediction event, the written log entry SHALL contain all
    required fields: timestamp (ISO 8601), match_id (non-empty string),
    predicted result, and model_weights snapshot.

    **Validates: Requirements 9.6**
    """

    @given(entry=prediction_log_entries)
    @settings(max_examples=100)
    def test_prediction_log_contains_all_required_fields(
        self, entry: PredictionLogEntry
    ) -> None:
        """
        For any PredictionLogEntry, after appending to the prediction log,
        the written entry must contain all required fields.

        **Validates: Requirements 9.6**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dm = DataManager(data_dir=tmp_path)

            # Write the prediction log entry
            dm.append_prediction_log(entry)

            # Read back the predictions log file
            log_file = tmp_path / "predictions_log.json"
            assert log_file.exists(), "predictions_log.json should exist after append"

            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            predictions = data.get("predictions", [])
            assert len(predictions) == 1, "Should have exactly one prediction entry"

            written_entry = predictions[0]

            # Required field: timestamp (ISO 8601)
            assert "timestamp" in written_entry, "Entry must contain 'timestamp'"
            assert isinstance(written_entry["timestamp"], str)
            assert len(written_entry["timestamp"]) > 0
            # Validate ISO 8601 format by parsing
            datetime.fromisoformat(written_entry["timestamp"])

            # Required field: match_id (non-empty string)
            assert "match_id" in written_entry, "Entry must contain 'match_id'"
            assert isinstance(written_entry["match_id"], str)
            assert len(written_entry["match_id"]) > 0

            # Required field: predicted result (predicted_score and win_draw_lose)
            assert "predicted_score" in written_entry, "Entry must contain 'predicted_score'"
            assert "win_draw_lose" in written_entry, "Entry must contain 'win_draw_lose'"

            # Required field: model_weights snapshot
            assert "model_weights" in written_entry, "Entry must contain 'model_weights'"
            assert isinstance(written_entry["model_weights"], dict)
