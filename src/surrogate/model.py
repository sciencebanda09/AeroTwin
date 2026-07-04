"""Serializable multi-output surrogate model."""

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.dataset.features import engineer_all_features
from src.uncertainty.conformal import ConformalRegressor


class SurrogateModel:
    """Column-aware wrapper around a fitted scikit-learn pipeline.

    ``feature_names`` remains the raw sensor schema (what every caller —
    ``DigitalTwin``, the trainer, the dashboard, the API — already passes
    in). Internally, ``fit``/``predict`` expand each row with
    ``engineer_all_features`` (ratios, deltas, and physics-residual health
    signals) before handing it to the wrapped sklearn pipeline, so no
    caller needs to change.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        feature_names: list[str],
        target_names: list[str],
        pipeline_feature_names: list[str] | None = None,
        calibrator: ConformalRegressor | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.target_names = target_names
        self.pipeline_feature_names = pipeline_feature_names or feature_names
        self.calibrator = calibrator

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Expand raw sensor rows with engineered features for the pipeline."""
        return engineer_all_features(frame[self.feature_names])

    def _postprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        """Apply output-domain constraints shared by all surrogate predictions."""
        out = values.copy()
        for column in out.columns:
            if column.endswith("Health"):
                out[column] = out[column].clip(0.0, 1.0)
            elif column in {"Thrust", "TSFC"}:
                out[column] = out[column].clip(lower=0.0)
        return out

    def fit(self, frame: pd.DataFrame) -> "SurrogateModel":
        """Fit inputs to all configured targets."""
        self.pipeline.fit(self._prepare(frame), frame[self.target_names])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return named predictions."""
        values = np.asarray(self.pipeline.predict(self._prepare(frame)))
        prediction = pd.DataFrame(values, columns=self.target_names, index=frame.index)
        return self._postprocess(prediction)

    def calibrate(self, frame: pd.DataFrame, coverage: float = 0.9) -> "SurrogateModel":
        """Fit split-conformal residual intervals for every target."""
        prediction = self.predict(frame)
        self.calibrator = ConformalRegressor(coverage).fit(
            frame[self.target_names].to_numpy(), prediction.to_numpy()
        )
        return self

    def predict_with_uncertainty(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
        """Return point predictions, lower/upper bounds, and calibrated coverage."""
        prediction = self.predict(frame)
        if self.calibrator is None:
            lower = prediction.copy()
            upper = prediction.copy()
            return prediction, lower, upper, 0.0
        lower_values, upper_values = self.calibrator.predict_interval(prediction.to_numpy())
        lower = self._postprocess(
            pd.DataFrame(lower_values, columns=self.target_names, index=frame.index)
        )
        upper = self._postprocess(
            pd.DataFrame(upper_values, columns=self.target_names, index=frame.index)
        )
        return prediction, lower, upper, self.calibrator.coverage

    def save(self, path: str | Path) -> None:
        """Atomically serialize the model payload."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        joblib.dump(self, temporary)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateModel":
        """Load and type-check a serialized model."""
        model: Any = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("artifact is not a SurrogateModel")
        if not hasattr(model, "pipeline_feature_names"):
            model.pipeline_feature_names = model.feature_names
        if not hasattr(model, "calibrator"):
            model.calibrator = None
        return model
