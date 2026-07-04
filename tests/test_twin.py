from pipeline import demo_data
from src.digital_twin.engine import DigitalTwin


def test_physics_fallback_and_persistence(tmp_path) -> None:
    frame = demo_data(1, 3)
    twin = DigitalTwin("E1")
    result = twin.batch_predict(frame)
    assert len(result) == 3
    assert {"Cycle", "Confidence", "ThrustLower", "ThrustUpper", "DegradationRate"}.issubset(
        result.columns
    )
    assert result["OverallHealth"].between(0, 1).all()
    assert result["OverallHealth"].is_monotonic_decreasing
    path = tmp_path / "state.json"
    twin.save_state(path)
    restored = DigitalTwin().load_state(path)
    assert len(restored.history) == 3


def test_health_estimation_is_monotonic() -> None:
    twin = DigitalTwin("E1")
    degraded = {
        "CompressorHealth": 0.5,
        "CombustorHealth": 0.5,
        "TurbineHealth": 0.5,
        "OverallHealth": 0.5,
        "Thrust": 1000.0,
        "TSFC": 0.001,
    }
    healthy = degraded | {
        "CompressorHealth": 1.0,
        "CombustorHealth": 1.0,
        "TurbineHealth": 1.0,
        "OverallHealth": 1.0,
    }
    first = twin.estimate_health({}, degraded)
    second = twin.estimate_health({}, healthy)
    assert second["OverallHealth"] <= first["OverallHealth"]
