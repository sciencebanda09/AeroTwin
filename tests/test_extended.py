from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset.loader import load_dataset
from src.digital_twin.fleet import rank_fleet
from src.digital_twin.runtime import DriftMonitor
from src.maintenance.recommendation import recommend
from src.maintenance.scheduler import ScheduleItem, schedule
from src.metrics.uncertainty import interval_metrics
from src.physics.component_maps import compressor_efficiency, turbine_efficiency
from src.surrogate.benchmark import benchmark
from src.training.evaluate import evaluate_model
from src.uncertainty.conformal import ConformalRegressor
from src.utils.config import load_config
from src.utils.timer import Timer
from src.viz.engine_animation import engine_schematic
from src.viz.plots import health_gauge, trend


def test_uncertainty_estimators() -> None:
    truth = np.array([[1.0], [2.0], [3.0]])
    pred = truth + 0.1
    conformal = ConformalRegressor(0.8).fit(truth, pred)
    lower, upper = conformal.predict_interval(pred)
    assert interval_metrics(truth, lower, upper)["coverage"] == 1
    with pytest.raises(RuntimeError):
        ConformalRegressor().predict_interval(pred)


def test_fleet_drift_and_maintenance() -> None:
    fleet = pd.DataFrame(
        {
            "engine_id": ["a", "b"],
            "OverallHealth": [0.9, 0.4],
            "FailureProbability": [0.01, 0.6],
            "RULCycles": [500, 20],
        }
    )
    assert rank_fleet(fleet).iloc[0]["engine_id"] == "b"
    monitor = DriftMonitor(threshold=0.01, window=2)
    assert not monitor.update(100, 110)
    assert monitor.update(100, 120)
    assert recommend(0.2, 5, 0.8).risk_level == "critical"
    assert recommend(0.8, 500, 0.01).risk_level == "low"
    items = [ScheduleItem("a", 10, 0.2), ScheduleItem("b", 20, 0.9)]
    assert schedule(items, 1)[0].engine_id == "b"


def test_component_metrics_and_constraints() -> None:
    assert 0 < compressor_efficiency(0.9) <= 1
    assert 0 < turbine_efficiency(0.9) <= 1


def test_reports_config_utilities_and_plots(tmp_path: Path) -> None:
    assert load_config().seed == 42
    with Timer() as timer:
        sum(range(10))
    assert timer.elapsed >= 0
    assert len(health_gauge(0.8).data) == 1
    assert len(engine_schematic({"CompressorHealth": 0.9}).data) == 1
    assert len(trend(pd.DataFrame({"Cycle": [1, 2], "x": [2, 3]}), ["x"]).data) == 1


def test_loader_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"EngineID": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required"):
        load_dataset(path)


def test_benchmark_cross_validation_and_evaluation(tmp_path: Path) -> None:
    from pipeline import demo_data
    from src.surrogate.train import create_model

    frame = demo_data(4, 8)
    train = frame[frame["EngineID"] <= 2]
    validation = frame[frame["EngineID"] > 2]
    best, results = benchmark(train, validation, kinds=("extra_trees",))
    assert results[0].kind == "extra_trees"
    assert len(best.predict(validation)) == len(validation)
    model_path = tmp_path / "model.joblib"
    create_model(n_estimators=10).fit(train).save(model_path)
    data_path = tmp_path / "data.csv"
    validation.to_csv(data_path, index=False)
    assert evaluate_model(model_path, data_path)["rmse"] >= 0
