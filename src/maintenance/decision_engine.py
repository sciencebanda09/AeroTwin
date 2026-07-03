"""Multi-option maintenance decision engine.

Where :mod:`src.maintenance.recommendation` returns a single recommended action,
this module generates the realistic menu of options a maintenance planner
actually chooses from (do nothing / monitor, inspect, repair, overhaul,
replace), each scored on cost, downtime, residual risk, and expected RUL gain,
then ranks them by a configurable weighted utility.

This module reuses :func:`src.maintenance.economics.estimate_economics` for
cost/risk math and does not replace :func:`src.maintenance.recommendation.recommend`;
both stay available for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

from src.maintenance.economics import estimate_economics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaintenanceOption:
    """One candidate maintenance action with its estimated tradeoffs."""

    action: str
    estimated_cost: float
    downtime_hours: float
    expected_risk: float
    expected_rul_gain_cycles: float
    failure_probability_after: float
    utility_score: float
    rationale: str


# (action, cost, downtime_hours, rul_gain_fraction_of_deficit, risk_reduction_fraction)
# rul_gain_fraction_of_deficit: fraction of the gap to a "full life" RUL horizon
# recovered by this action. risk_reduction_fraction: fraction by which the
# action reduces failure probability.
_OPTION_TEMPLATE = (
    ("Continue operation / monitor", 0.0, 0.0, 0.0, 0.0),
    ("Increase monitoring frequency", 1_500.0, 0.0, 0.05, 0.05),
    ("Borescope inspection", 8_000.0, 4.0, 0.15, 0.20),
    ("Targeted component repair", 45_000.0, 24.0, 0.55, 0.65),
    ("Full overhaul", 180_000.0, 96.0, 0.95, 0.92),
    ("Remove and replace engine", 750_000.0, 168.0, 1.0, 0.99),
)


class MaintenanceDecisionEngine:
    """Generates and ranks maintenance options for a given engine state."""

    def __init__(
        self,
        cost_weight: float = 0.3,
        downtime_weight: float = 0.2,
        risk_weight: float = 0.35,
        rul_gain_weight: float = 0.15,
        full_life_horizon_cycles: float = 300.0,
        failure_cost: float = 500_000.0,
    ) -> None:
        """Create the engine.

        Args:
            cost_weight: Utility weight penalizing estimated cost (higher cost
                = lower utility).
            downtime_weight: Utility weight penalizing downtime hours.
            risk_weight: Utility weight rewarding residual-risk reduction.
            rul_gain_weight: Utility weight rewarding expected RUL gain.
            full_life_horizon_cycles: Reference "as-good-as-new" RUL horizon
                used to scale RUL gain estimates.
            failure_cost: Cost of an unplanned failure, passed through to
                :func:`estimate_economics`.
        """
        total = cost_weight + downtime_weight + risk_weight + rul_gain_weight
        if total <= 0:
            raise ValueError("at least one weight must be positive")
        self.cost_weight = cost_weight / total
        self.downtime_weight = downtime_weight / total
        self.risk_weight = risk_weight / total
        self.rul_gain_weight = rul_gain_weight / total
        self.full_life_horizon_cycles = full_life_horizon_cycles
        self.failure_cost = failure_cost

    def generate_options(
        self, health: float, rul_cycles: float, failure_probability: float
    ) -> list[MaintenanceOption]:
        """Generate every candidate maintenance option, ranked best-first.

        Args:
            health: Current overall health in ``[0, 1]``.
            rul_cycles: Current predicted remaining useful life, in cycles.
            failure_probability: Current failure probability in ``[0, 1]``.

        Returns:
            Options sorted descending by ``utility_score``.
        """
        if not 0.0 <= health <= 1.0:
            raise ValueError(f"health must be in [0, 1], got {health}")
        if not 0.0 <= failure_probability <= 1.0:
            raise ValueError(f"failure_probability must be in [0, 1], got {failure_probability}")
        if rul_cycles < 0:
            raise ValueError("rul_cycles must be nonnegative")

        rul_deficit = max(self.full_life_horizon_cycles - rul_cycles, 0.0)
        costs = [max(c, 1.0) for _, c, _, _, _ in _OPTION_TEMPLATE]
        downtimes = [max(d, 1.0) for _, _, d, _, _ in _OPTION_TEMPLATE]
        max_cost, max_downtime = max(costs), max(downtimes)

        options: list[MaintenanceOption] = []
        for (action, cost, downtime, rul_frac, risk_frac), norm_cost, norm_downtime in zip(
            _OPTION_TEMPLATE, costs, downtimes, strict=True
        ):
            economics = estimate_economics(
                failure_probability, planned_cost=cost, failure_cost=self.failure_cost,
                downtime_hours=downtime,
            )
            rul_gain = rul_frac * rul_deficit
            risk_after = failure_probability * (1.0 - risk_frac)
            utility = (
                self.cost_weight * (1.0 - norm_cost / max_cost)
                + self.downtime_weight * (1.0 - norm_downtime / max_downtime)
                + self.risk_weight * (failure_probability - risk_after)
                + self.rul_gain_weight * (rul_gain / max(rul_deficit, 1.0))
            )
            rationale = (
                f"Reduces failure probability from {failure_probability:.2f} to {risk_after:.2f}, "
                f"gains ~{rul_gain:.0f} RUL cycles, costs {economics.planned_cost:,.0f} "
                f"with {downtime:.0f}h downtime."
            )
            options.append(
                MaintenanceOption(
                    action=action,
                    estimated_cost=economics.planned_cost,
                    downtime_hours=downtime,
                    expected_risk=round(risk_after, 4),
                    expected_rul_gain_cycles=round(rul_gain, 2),
                    failure_probability_after=round(risk_after, 4),
                    utility_score=round(utility, 4),
                    rationale=rationale,
                )
            )
        options.sort(key=lambda option: option.utility_score, reverse=True)
        logger.info(
            "generated %d maintenance options; top choice=%s", len(options), options[0].action
        )
        return options

    def recommend_top(
        self, health: float, rul_cycles: float, failure_probability: float
    ) -> MaintenanceOption:
        """Return the single highest-utility option."""
        return self.generate_options(health, rul_cycles, failure_probability)[0]
