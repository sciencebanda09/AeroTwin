"""Condition-based maintenance orchestration."""

from .recommendation import MaintenanceDecision, recommend


def assess(
    health: float, rul_cycles: float, probability: float, subsystem_health: dict[str, float]
) -> MaintenanceDecision:
    """Identify weakest subsystem and create a maintenance decision."""
    weakest = (
        min(subsystem_health, key=lambda k: subsystem_health[k]) if subsystem_health else "engine"
    )
    return recommend(health, rul_cycles, probability, weakest)
