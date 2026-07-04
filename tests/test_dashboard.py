from pathlib import Path


def test_dashboard_surfaces_required_twin_outputs() -> None:
    source = Path("src/viz/dashboard.py").read_text(encoding="utf-8")
    for field in (
        "Altitude",
        "Mach",
        "RPM",
        "Fuel flow",
        "CompressorHealth",
        "CombustorHealth",
        "TurbineHealth",
        "OverallHealth",
        "Thrust",
        "DegradationRate",
        "Confidence",
    ):
        assert field in source
