import pytest
from jsonschema import ValidationError

from ats.contracts import (
    CANONICAL_CLASSES,
    format_answer_text,
    validate_answer,
    validate_cost_report,
    validate_question_set,
    validate_window_track,
)


def test_canonical_class_order():
    assert CANONICAL_CLASSES == (
        "LYING",
        "SITTING",
        "STANDING_STILL",
        "STANDING_MOVING",
        "WALKING",
        "RUNNING",
        "BICYCLING",
    )


def test_window_track_valid_example_round_trips():
    entry = {
        "window_id": "w000123",
        "t_start": 905.0,
        "t_end": 915.0,
        "probs": [0.01, 0.02, 0.02, 0.03, 0.85, 0.05, 0.02],
        "coverage": 1.0,
        "feature_summary": {
            "acc_mag_mean": 10.2,
            "acc_mag_std": 1.4,
            "dominant_cadence_hz": 1.8,
            "gyro_energy_x": 0.4,
            "gyro_energy_y": 0.3,
            "gyro_energy_z": 0.1,
        },
        "model_id": "oracle",
    }
    validate_window_track(entry)


def test_window_track_missing_required_field_is_rejected():
    entry = {
        "window_id": "w000123",
        "t_start": 905.0,
        "t_end": 915.0,
        "probs": [0.01, 0.02, 0.02, 0.03, 0.85, 0.05, 0.02],
        "coverage": 1.0,
        "model_id": "oracle",
    }
    with pytest.raises(ValidationError):
        validate_window_track(entry)


def test_window_track_wrong_prob_length_is_rejected():
    entry = {
        "window_id": "w000123",
        "t_start": 905.0,
        "t_end": 915.0,
        "probs": [0.5, 0.5],
        "coverage": 1.0,
        "feature_summary": {
            "acc_mag_mean": 10.2,
            "acc_mag_std": 1.4,
            "dominant_cadence_hz": 1.8,
            "gyro_energy_x": 0.4,
            "gyro_energy_y": 0.3,
            "gyro_energy_z": 0.1,
        },
        "model_id": "oracle",
    }
    with pytest.raises(ValidationError):
        validate_window_track(entry)


def _valid_answer():
    return {
        "question_id": "q001",
        "answer": "700 seconds",
        "activity_event": "Walking",
        "evidence": {
            "timestamps": "905 to 1420, 2110 to 2295 (seconds from start)",
            "sensor_modality": "Accelerometer, Gyroscope",
            "sensor_channels": "All",
        },
        "explanation": "Walking was detected in two separate intervals, of 515 and 185 seconds, which sum to 700 seconds.",
        "tier_inferred": 2,
        "cited_intervals": [[905.0, 1420.0], [2110.0, 2295.0]],
        "modality": "both",
        "channels": ["all"],
    }


def test_answer_valid_example_round_trips():
    validate_answer(_valid_answer())


def test_answer_rejects_unknown_modality():
    answer = _valid_answer()
    answer["modality"] = "video"
    with pytest.raises(ValidationError):
        validate_answer(answer)


def test_format_answer_text_field_order():
    text = format_answer_text(_valid_answer())
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0].startswith("Answer:")
    assert lines[1].startswith("Activity/Event:")
    assert lines[2].startswith("Evidence:")
    assert "Timestamp(s):" in lines[3]
    assert "Sensor Modality:" in lines[4]
    assert "Sensor Channel(s):" in lines[5]
    assert lines[6].startswith("Explanation:")


def test_question_set_with_gold_round_trips():
    question_set = {
        "questions": [
            {
                "question_id": "q001",
                "text": "How long was the user walking?",
                "gold": {
                    "answer": "700 seconds",
                    "activity_event": "Walking",
                    "cited_intervals": [[905.0, 1420.0], [2110.0, 2295.0]],
                    "modality": "both",
                    "channels": ["all"],
                },
            }
        ]
    }
    validate_question_set(question_set)


def test_question_set_without_gold_round_trips():
    """Eval-time input carries no gold block and must still validate cleanly,
    so it takes the exact same code path as our dev question sets."""
    question_set = {
        "questions": [
            {"question_id": "q001", "text": "What activity is the user performing?"}
        ]
    }
    validate_question_set(question_set)


def test_cost_report_valid_example_round_trips():
    report = {
        "config_id": "full",
        "target_device": "Laptop CPU: AMD Ryzen 7 5800H (8C/16T), 16GB RAM, Windows 11 64-bit, single-process CPU inference",
        "params": 1_500_000,
        "disk_mb": 6.2,
        "peak_rss_mb": 210.5,
        "latency_p50_ms": 45.0,
        "latency_p95_ms": 78.0,
    }
    validate_cost_report(report)


def test_cost_report_missing_required_field_is_rejected():
    report = {
        "config_id": "full",
        "params": 1_500_000,
        "disk_mb": 6.2,
        "peak_rss_mb": 210.5,
        "latency_p50_ms": 45.0,
        "latency_p95_ms": 78.0,
    }
    with pytest.raises(ValidationError):
        validate_cost_report(report)
