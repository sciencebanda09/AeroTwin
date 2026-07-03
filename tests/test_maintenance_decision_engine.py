"""Unit tests for src.maintenance.decision_engine."""

import pytest

from src.maintenance.decision_engine import MaintenanceDecisionEngine


def test_generates_all_options_sorted_by_utility() -> None:
    engine = MaintenanceDecisionEngine()
    options = engine.generate_options(health=0.9, rul_cycles=250, failure_probability=0.02)
    assert len(options) == 6
    scores = [option.utility_score for option in options]
    assert scores == sorted(scores, reverse=True)


def test_healthy_engine_prefers_low_cost_option() -> None:
    engine = MaintenanceDecisionEngine()
    top = engine.recommend_top(health=0.95, rul_cycles=290, failure_probability=0.01)
    assert top.estimated_cost <= 50_000


def test_critical_engine_prefers_aggressive_option() -> None:
    engine = MaintenanceDecisionEngine()
    top = engine.recommend_top(health=0.2, rul_cycles=5, failure_probability=0.9)
    assert top.action in {"Full overhaul", "Remove and replace engine", "Targeted component repair"}


def test_invalid_health_raises() -> None:
    engine = MaintenanceDecisionEngine()
    with pytest.raises(ValueError):
        engine.generate_options(health=1.5, rul_cycles=100, failure_probability=0.1)


def test_invalid_weights_raise() -> None:
    with pytest.raises(ValueError):
        MaintenanceDecisionEngine(0, 0, 0, 0)


def test_rul_gain_nonnegative() -> None:
    engine = MaintenanceDecisionEngine()
    options = engine.generate_options(health=0.5, rul_cycles=100, failure_probability=0.3)
    assert all(option.expected_rul_gain_cycles >= 0 for option in options)
