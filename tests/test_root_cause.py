"""Unit tests for src.explainability.root_cause."""

from src.explainability.root_cause import analyze_faults, analyze_scenario


def test_analyze_scenario_ranks_dominant_factor() -> None:
    baseline = {"FuelFlow": 1.0, "RPM": 80_000.0, "Tamb": 288.0, "Pamb": 101_325.0}
    adjusted = {"FuelFlow": 2.0, "RPM": 80_000.0, "Tamb": 288.0, "Pamb": 101_325.0}
    report = analyze_scenario(baseline, adjusted, health_delta=-0.1)
    assert report.factors
    assert report.factors[0].factor == "FuelFlow"
    assert "decreased" in report.summary


def test_analyze_scenario_no_change_yields_no_factors() -> None:
    baseline = {"FuelFlow": 1.0, "RPM": 80_000.0}
    report = analyze_scenario(baseline, dict(baseline), health_delta=0.0)
    assert report.factors == []
    assert "did not change" in report.summary


def test_analyze_faults_ranks_by_severity() -> None:
    fault_summary = [
        {"fault_type": "compressor_fouling", "severity": 0.2},
        {"fault_type": "turbine_erosion", "severity": 0.8},
    ]
    report = analyze_faults(fault_summary, health_delta=-0.15)
    assert report.factors[0].factor == "turbine_erosion"
    assert len(report.causal_chain) >= 2
