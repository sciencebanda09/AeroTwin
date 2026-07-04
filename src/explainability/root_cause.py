"""Root cause analysis for health and RUL predictions.

Turns a before/after comparison (typically from
:class:`src.simulation.what_if.ScenarioSimulator` or a fault-injection run) into
a ranked, human-readable causal chain: which inputs changed, how much each
contributed to the health delta, and the resulting narrative.

If a trained surrogate model with a tree-based estimator is loaded and the
optional ``shap`` package is installed, SHAP values are used to rank
contributions for ML-model predictions. Otherwise a physics-based sensitivity
ranking is used, which requires no extra dependency and always works for the
physics fallback path.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import shap

    _HAS_SHAP = True
except ImportError:  # pragma: no cover
    shap = None
    _HAS_SHAP = False


# Approximate physics sensitivity of overall health to each named input,
# expressed as (health-change-per-unit-relative-change, causal narrative).
# These coefficients reflect the qualitative physics in
# src.physics.cycle_model.BraytonCycle: higher fuel flow raises turbine inlet
# temperature and thermal stress (hurting turbine/combustor health headroom
# over time); lower compressor/turbine efficiency directly reduces health.
_SENSITIVITY = {
    "FuelFlow": (
        0.6,
        "Fuel flow increased -> turbine inlet temperature rose -> thermal stress increased",
    ),
    "RPM": (0.4, "RPM increased -> compressor pressure ratio rose -> mechanical loading increased"),
    "Tamb": (0.15, "Ambient temperature increased -> compressor work increased -> margin reduced"),
    "Pamb": (0.1, "Ambient pressure changed -> station pressures shifted -> cycle margin shifted"),
    "compressor_efficiency": (
        1.0,
        "Compressor efficiency decreased -> pressure ratio degraded -> compressor health decreased",
    ),
    "turbine_efficiency": (
        1.0,
        "Turbine efficiency decreased -> expansion work degraded -> turbine health decreased",
    ),
}


@dataclass(frozen=True)
class ContributingFactor:
    """One ranked cause behind an observed prediction change."""

    factor: str
    contribution: float
    explanation: str


@dataclass(frozen=True)
class RootCauseReport:
    """Ranked explanation for a health/RUL change."""

    summary: str
    factors: list[ContributingFactor]
    causal_chain: list[str]


def _relative_change(before: float | None, after: float | None) -> float:
    """Return the signed relative change of after vs before, guarding zero."""
    if before is None or after is None:
        return 0.0
    denom = abs(before) if abs(before) > 1e-9 else 1.0
    return (after - before) / denom


def analyze_scenario(
    baseline_inputs: dict[str, float],
    adjusted_inputs: dict[str, float],
    health_delta: float,
) -> RootCauseReport:
    """Explain a what-if health change via physics-sensitivity ranking.

    Args:
        baseline_inputs: Input values before adjustment (e.g. FuelFlow, RPM,
            Tamb, Pamb, compressor_efficiency, turbine_efficiency).
        adjusted_inputs: Same keys, after adjustment.
        health_delta: ``adjusted_health - baseline_health`` from the simulation.

    Returns:
        Ranked contributing factors and a causal-chain narrative.
    """
    scored: list[ContributingFactor] = []
    for key, (weight, narrative) in _SENSITIVITY.items():
        if key not in baseline_inputs or key not in adjusted_inputs:
            continue
        rel_change = _relative_change(baseline_inputs[key], adjusted_inputs[key])
        if abs(rel_change) < 1e-6:
            continue
        contribution = weight * rel_change
        scored.append(ContributingFactor(key, round(contribution, 4), narrative))

    scored.sort(key=lambda item: abs(item.contribution), reverse=True)
    direction = (
        "decreased" if health_delta < 0 else "increased" if health_delta > 0 else "did not change"
    )
    summary = f"Overall health {direction} by {abs(health_delta):.3f}."
    if scored:
        summary += (
            f" Primary driver: {scored[0].factor} ({scored[0].explanation.split(' -> ')[0]})."
        )
    chain = [f.explanation for f in scored[:3]]
    if chain:
        chain.append("-> RUL and failure probability updated accordingly")
    logger.info("root cause analysis: %d factors ranked", len(scored))
    return RootCauseReport(summary, scored, chain)


def analyze_faults(fault_summary: list[dict[str, Any]], health_delta: float) -> RootCauseReport:
    """Explain a health change driven by active faults (see :mod:`src.faults.injection`).

    Args:
        fault_summary: Output of ``FaultInjector.to_summary()``.
        health_delta: Observed health change attributable to the faults.

    Returns:
        Ranked contributing factors, one per active fault, by severity.
    """
    narratives = {
        "compressor_fouling": "Compressor fouling -> pressure ratio degraded -> compressor health decreased",
        "turbine_erosion": "Turbine erosion -> expansion efficiency degraded -> turbine health decreased",
        "fuel_nozzle_blockage": "Fuel nozzle blockage -> fuel delivery restricted -> combustion energy decreased",
        "bearing_wear": "Bearing wear -> spool friction increased -> compressor and turbine health decreased",
        "sensor_drift": "Sensor drift -> measurement bias grew over time -> estimated health distorted",
        "sensor_bias": "Sensor bias -> fixed measurement offset -> estimated health distorted",
    }
    scored = sorted(
        (
            ContributingFactor(
                item["fault_type"],
                round(float(item.get("severity", 0.0)), 4),
                narratives.get(item["fault_type"], "Fault active -> health impacted"),
            )
            for item in fault_summary
        ),
        key=lambda item: item.contribution,
        reverse=True,
    )
    direction = (
        "decreased" if health_delta < 0 else "increased" if health_delta > 0 else "did not change"
    )
    summary = f"Overall health {direction} by {abs(health_delta):.3f} due to {len(scored)} active fault(s)."
    chain = [f.explanation for f in scored[:3]]
    if chain:
        chain.append("-> RUL decreased -> maintenance urgency increased")
    return RootCauseReport(summary, scored, chain)


def shap_feature_importance(model: Any, frame: Any) -> list[ContributingFactor] | None:
    """Rank ML-model feature contributions with SHAP, if available.

    Args:
        model: A loaded ``SurrogateModel`` (see ``src.surrogate.model``) whose
            underlying estimator SHAP supports (tree-based models).
        frame: A single-row ``pandas.DataFrame`` of feature values matching
            ``model.feature_names``.

    Returns:
        Ranked factors by mean absolute SHAP value, or ``None`` if SHAP is not
        installed or the estimator is unsupported.
    """
    if not _HAS_SHAP:
        logger.info("shap not installed; skipping SHAP-based root cause ranking")
        return None
    pipeline = getattr(model, "pipeline", None)
    estimator = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
    try:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(frame)
    except Exception as error:  # pragma: no cover - depends on optional estimator support
        logger.warning("SHAP explanation failed: %s", error)
        return None
    row = values[0] if hasattr(values, "__len__") and len(values) else values
    factors = [
        ContributingFactor(name, round(float(val), 4), f"{name} contributed via learned model")
        for name, val in zip(model.feature_names, row, strict=False)
    ]
    factors.sort(key=lambda item: abs(item.contribution), reverse=True)
    return factors
