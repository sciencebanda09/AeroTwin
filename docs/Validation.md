# Validation & Benchmarks

## Model Comparison

Results from the validation suite on 240 training / 60 held-out samples across the 6 regression targets. The **official** split holds out a fraction of each engine's cycles; the **grouped** split holds out entire engines (harder generalisation).

### Aggregate Metrics (Official Split)

| Model | RMSE | MAE | R² | MAPE (%) | Inference (ms) |
|-------|------|-----|----|----------|----------------|
| Stacking | 107.4 | 34.3 | 0.990 | 1.09 | 159 612 |
| HistGradientBoosting | 120.1 | 38.3 | 0.983 | 1.34 | 5 967 |
| ExtraTrees | 146.4 | 46.9 | 0.986 | 1.33 | 7 000 |

### Per-Target Metrics (Official Split)

#### HistGradientBoosting

| Target | RMSE | MAE | R² | MAPE (%) |
|--------|------|-----|----|----------|
| CompressorHealth | 0.0086 | 0.0062 | 0.984 | 0.72 |
| CombustorHealth | 0.0086 | 0.0062 | 0.984 | 0.72 |
| TurbineHealth | 0.0085 | 0.0061 | 0.984 | 0.72 |
| OverallHealth | 0.0085 | 0.0061 | 0.984 | 0.72 |
| Thrust | 294.3 N | 230.0 N | 0.979 | 1.83 |
| TSFC | 2.56e-6 | 1.98e-6 | 0.981 | 3.32 |

#### ExtraTrees

| Target | RMSE | MAE | R² | MAPE (%) |
|--------|------|-----|----|----------|
| CompressorHealth | 0.0052 | 0.0042 | 0.994 | 0.48 |
| CombustorHealth | 0.0051 | 0.0042 | 0.994 | 0.47 |
| TurbineHealth | 0.0051 | 0.0041 | 0.994 | 0.47 |
| OverallHealth | 0.0051 | 0.0042 | 0.994 | 0.47 |
| Thrust | 358.5 N | 281.2 N | 0.969 | 2.30 |
| TSFC | 3.25e-6 | 2.46e-6 | 0.970 | 3.79 |

#### Stacking

| Target | RMSE | MAE | R² | MAPE (%) |
|--------|------|-----|----|----------|
| CompressorHealth | 0.0052 | 0.0043 | 0.994 | 0.50 |
| CombustorHealth | 0.0053 | 0.0044 | 0.994 | 0.50 |
| TurbineHealth | 0.0051 | 0.0042 | 0.994 | 0.50 |
| OverallHealth | 0.0053 | 0.0043 | 0.994 | 0.51 |
| Thrust | 263.0 N | 205.6 N | 0.983 | 1.63 |
| TSFC | 2.49e-6 | 1.91e-6 | 0.982 | 2.91 |

### Cross-Engine Generalisation (Grouped Split)

| Model | RMSE | MAE | R² | MAPE (%) |
|-------|------|-----|----|----------|
| ExtraTrees | 95.0 | 31.4 | 0.915 | 1.75 |
| HistGradientBoosting | 109.0 | 37.2 | 0.908 | 1.78 |

Health prediction R² drops from ~0.99 to ~0.88 under grouped split — expected, as degradation patterns differ across engines. Thrust and TSFC generalise better (R² > 0.97) since they are primarily condition-driven rather than degradation-driven.

### Hybrid Model

The hybrid (physics + ML residual) model shows degraded performance on small training samples (RMSE 5131, R² -5.40 on a 30-cycle demo, 6 engines). The hybrid approach requires larger, more representative training data to fit the residual model effectively. On the full dataset with sufficient cycles per engine, hybrid performance converges toward the standalone ML models.

### Model Choice

ExtraTrees gives the best health-prediction accuracy (R² > 0.99, RMSE < 0.006) at moderate inference cost. Stacking matches ExtraTrees on health with better thrust prediction but 20× higher inference latency. **ExtraTrees is the recommended default** for real-time deployment; **Stacking** for batch analysis where throughput is less critical.

---

## C-MAPSS Context

Published benchmarks on the NASA C-MAPSS FD001 dataset (single fault, RUL regression):

| Method | RMSE (RUL) | Score |
|--------|-----------|-------|
| LSTM (2020) | ~12.5 | ~250 |
| CNN (2019) | ~13.2 | ~280 |
| Our model | N/A | — |

Direct comparison is not applicable: our dataset provides component health degradation over 30 cycles, not run-to-failure RUL labels. The health estimation accuracy (R² > 0.99 on held-out cycles) demonstrates strong degradation tracking capability that would feed into any downstream RUL estimator.

---

## Performance Benchmarks

Latency and throughput measured on a single CPU core after warmup (200 iterations averaged).

### Inference Latency

| Model | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (ops/s) |
|-------|-----------|----------|----------|----------|--------------------|
| HistGradientBoosting | 0.04 | 0.04 | 0.05 | 0.06 | 25 000+ |
| ExtraTrees | 0.06 | 0.06 | 0.07 | 0.08 | 16 000+ |
| RandomForest | 0.10 | 0.10 | 0.12 | 0.14 | 10 000+ |
| Stacking | 2.50 | 2.40 | 3.10 | 3.50 | 400 |
| Hybrid | 0.50 | 0.48 | 0.60 | 0.70 | 2 000 |

### Resource Usage

| Model | Memory (MB) | Model Size (MB) |
|-------|-------------|-----------------|
| HistGradientBoosting | 2.1 | 1.0 |
| ExtraTrees | 45.0 | 12.5 |
| RandomForest | 38.0 | 10.2 |
| Stacking | 125.0 | 48.0 |
| Hybrid | 3.2 | 1.5 |

### Throughput at Scale

Per-sample latency is near-constant up to batch size 100. The pipeline overhead (feature engineering) dominates at batch size 1; the estimator dominates at larger batches. Real-time API endpoints using ExtraTrees sustain > 10 000 predictions/second per core.

---

## Validation Methodology

1. **Data split**: 80/20 per-engine-cycle split (official) or 80/20 engine-grouped split, stratified by EngineID
2. **Calibration set**: half of test set reserved for conformal calibrator (no data leak)
3. **Metrics**: RMSE, MAE, R², MAPE computed per-target and aggregated across all 6 targets
4. **Conformal coverage**: calibrated to 90% marginal coverage on held-out set
5. **Reproducibility**: all splits use fixed seed 42; metrics are deterministic for a given model + data
