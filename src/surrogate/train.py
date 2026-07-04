"""Surrogate construction and training."""

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from src.dataset.features import RESIDUAL_COLUMNS
from src.dataset.loader import FEATURES, TARGETS
from src.dataset.preprocess import build_preprocessor
from .model import SurrogateModel

# Ratio/delta columns added by engineer_features(), plus physics-residual
# health columns added by healthy_reference_residuals(). Must match what
# SurrogateModel._prepare() -> engineer_all_features() actually produces.
_ENGINEERED_COLUMNS = [
    "CompressorPR",
    "TurbinePR",
    "CompressorDeltaT",
    "TurbineDeltaT",
    "FuelPerRPM",
    "CorrectedRPM",
    *RESIDUAL_COLUMNS,
]
PIPELINE_FEATURES = FEATURES + _ENGINEERED_COLUMNS


def create_model(
    kind: str = "extra_trees", seed: int = 42, n_estimators: int = 200
) -> SurrogateModel:
    """Construct a reproducible multi-output surrogate."""
    gradient_boosting = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=n_estimators, random_state=seed)
    )
    estimators = {
        "extra_trees": ExtraTreesRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=2
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=2
        ),
        "gradient_boosting": gradient_boosting,
        "ensemble": MultiOutputRegressor(
            VotingRegressor(
                [
                    (
                        "extra_trees",
                        ExtraTreesRegressor(
                            n_estimators=n_estimators,
                            random_state=seed,
                            n_jobs=-1,
                            min_samples_leaf=2,
                        ),
                    ),
                    (
                        "random_forest",
                        RandomForestRegressor(
                            n_estimators=n_estimators,
                            random_state=seed,
                            n_jobs=-1,
                            min_samples_leaf=2,
                        ),
                    ),
                    (
                        "gradient_boosting",
                        GradientBoostingRegressor(n_estimators=n_estimators, random_state=seed),
                    ),
                ]
            )
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed, early_stopping=True
        ),
    }
    if kind == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise RuntimeError("Install xgboost to use model kind 'xgboost'") from error
        estimators[kind] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=n_estimators,
                max_depth=6,
                random_state=seed,
                objective="reg:squarederror",
            )
        )
    if kind not in estimators:
        raise ValueError(f"Unsupported model kind: {kind}")
    pipeline = Pipeline(
        [("preprocess", build_preprocessor(PIPELINE_FEATURES)), ("model", estimators[kind])]
    )
    # SurrogateModel.feature_names stays the raw schema; the pipeline itself
    # is fit on the wider engineered column set via SurrogateModel._prepare.
    return SurrogateModel(pipeline, FEATURES, TARGETS, PIPELINE_FEATURES)
