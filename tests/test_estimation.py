import numpy as np
import pytest

from src.estimation.ekf import ExtendedKalmanFilter
from src.estimation.state_estimator import StateEstimator


def test_ekf_converges() -> None:
    ekf = ExtendedKalmanFilter(np.array([0.0]), np.eye(1), np.eye(1) * 0.01, np.eye(1) * 0.1)
    for _ in range(20):
        ekf.predict(lambda x: x, np.eye(1))
        ekf.update(np.array([1.0]), lambda x: x, np.eye(1))
    assert ekf.state[0] == pytest.approx(1, abs=0.05)


def test_state_estimator_monotonic() -> None:
    estimator = StateEstimator()
    s1 = estimator.update(np.array([0.9, 0.9, 0.9, 0.9]))
    s2 = estimator.update(np.array([0.95, 0.95, 0.95, 0.95]))
    for i in range(4):
        assert s2[i] <= s1[i] + 1e-6, f"health component {i} increased (monotonicity violation)"
    assert all(0 <= s2[i] <= 1 for i in range(4)), "health out of [0, 1] range"


def test_state_estimator_no_cycle_leakage() -> None:
    estimator = StateEstimator()
    with pytest.raises((ValueError, TypeError, IndexError)):
        estimator.update(np.array([0.9, 0.9, 0.9, 0.9, 1.0]))
