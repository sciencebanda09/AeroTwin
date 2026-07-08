# Architecture

## Data Flow

```
Sensor Record (EngineID, Cycle, Altitude, Mach, Tamb, Pamb, RPM, FuelFlow, P2-T4)
    |
    +--> Schema / Range Validation
    |
    +--> [Feature Engineering]  -->  34 total features
    |     8 raw sensors
    |   + 6 physics-normalized residuals
    |   + 10 engineered ratios/deltas
    |
    +--> [Brayton-Cycle Physics Model]
    |     Variable specific heats, ISA atmosphere, 4th-order component maps
    |
    +--> [Learned Surrogate / Hybrid Physics+ML]
    |     Model kinds: HGBT, ExtraTrees, RandomForest, GradientBoosting,
    |     Stacking, XGBoost, MLP, Hybrid (physics + ML residual)
    |
    +--> [Bayesian State Estimator]
    |      EKF or UKF
    |      State: [health, degradation_rate]
    |      Observation: surrogate prediction
    |
    +--> DigitalTwin Facade
         |
         +--> RUL
         +--> Failure Probability
         +--> Health Trajectories
         +--> Fleet Ranking
          +--> FastAPI (8 endpoints)
          +--> Streamlit Dashboard (18 pages + 3D engine view)
          +--> Report Generator (Markdown)

### 3D Engine Visualization Pipeline

```
CAD STEP files (.zip source)
    |
    +--> [scripts/convert_engine_cad.py]
    |      --model generic_turbine | kj66
    |      Per-file loader (cadquery) for generic_turbine
    |         or XCAF/OCAF assembly reader (STEPCAFControl_Reader) for kj66
    |      Groups parts into: compressor / combustor / turbine / casing
    |
    +--> [vtkDecimatePro]  ~50 % triangle reduction
    |
    +--> models/engine_meshes/<model_name>/
    |      compressor.vtp, combustor.vtp, turbine.vtp, casing.vtp
    |
    +--> [src/viz/engine_3d.py]
    |      load_engine_meshes() -> dict[str, pv.PolyData]
    |      build_interactive_html() -> HTML with pyvista-jupyter viewer
    |      render_static_image() -> PNG for dashboard thumbnails
    |
    +--> Streamlit "3D Engine" view mode, health-coloured per stage
```

## Module Dependencies

```
src/physics         <-  src/surrogate/hybrid
src/dataset         ->  src/surrogate
src/surrogate       ->  src/uncertainty
src/surrogate       ->  src/explainability
src/digital_twin    ->  src/estimation
src/digital_twin    ->  src/prediction
src/simulation      ->  src/physics
src/validation      ->  src/surrogate
src/viz/engine_3d   ->  configs/viz_config.yaml
scripts/convert_engine_cad ->  configs/cad_models/*.yaml
pipeline.py         ->  all modules
```

## Design Decisions

1. **Hybrid Physics + ML** - ML learns the residual: `prediction = physics + ml_residual`. Physics handles condition-dependent variation; ML models the degradation signal.

2. **Target Scaling** - StandardScaler per target. Thrust (0-90 kN) and Health (0-1) have different scales; per-target metrics avoid misleading aggregates.

3. **Three Uncertainty Modes** - conformal (fast, marginal coverage), quantile (medium, conditional coverage), ensemble (slow, approximate).

4. **Data-Calibrated Failure Probability** - Logistic regression fitted on training degradation trajectories.

5. **Configurable RUL Thresholds** - RULConfig dataclass with configurable failure/warning thresholds.

6. **Stateful API** - Each engine has an in-memory DigitalTwin with Kalman state. REST `/update` and `/batch` both advance the same estimator.

## CLI Commands

| Command | Entry Point | Description |
|---------|-------------|-------------|
| `train` | `pipeline.py` | Train one model variant |
| `tune` | `pipeline.py` | Grid-search hyperparameters |
| `evaluate` | `pipeline.py` | Evaluate saved model |
| `predict` | `pipeline.py` | Batch inference |
| `experiment` | `pipeline.py` | Logged experiment run |
| `ablation` | `pipeline.py` | Cross-model ablation |
| `report` | `pipeline.py` | Markdown report generation |
| `validation` | `pipeline.py` | Cross-model validation suite |
| `benchmark` | `pipeline.py` | Latency/throughput benchmarks |
| `orchestrate` | `pipeline.py` | Train all, validate, benchmark |
| `demo` | `pipeline.py` | Demo with real data slice |
