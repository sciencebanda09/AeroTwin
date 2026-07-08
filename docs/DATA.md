# Dataset Contract

## Schema

One row represents one engine cycle. SI units: metres, kelvin, pascals,
rev/min, kg/s, newtons, kg/(N-s). Health values dimensionless [0, 1].

### Input Columns (14)

| Column | Unit | Description |
|--------|------|-------------|
| EngineID | - | Engine identifier |
| Cycle | - | Cycle number for this engine |
| Altitude | m | Flight altitude |
| Mach | - | Flight Mach number |
| Tamb | K | Ambient temperature |
| Pamb | Pa | Ambient pressure |
| RPM | rev/min | Engine shaft speed |
| FuelFlow | kg/s | Fuel mass flow rate |
| P2 | Pa | Compressor exit pressure |
| T2 | K | Compressor exit temperature |
| P3 | Pa | Combustor exit pressure |
| T3 | K | Combustor exit temperature |
| P4 | Pa | Turbine exit pressure |
| T4 | K | Turbine exit temperature |

### Target Columns (6)

| Column | Unit | Description |
|--------|------|-------------|
| CompressorHealth | - | Compressor health [0, 1] |
| CombustorHealth | - | Combustor health [0, 1] |
| TurbineHealth | - | Turbine health [0, 1] |
| OverallHealth | - | Fused health [0, 1] |
| Thrust | N | Engine thrust |
| TSFC | kg/(N-s) | Thrust-specific fuel consumption |

## Feature Engineering

Feature engineering (`src/dataset/features.py`) adds 20 derived features,
producing 34 total features used by the surrogate pipeline.

**Physics Residuals (6):** ResP2, ResT2, ResP3, ResT3, ResP4, ResT4
- Fractional deviation of measured station values from healthy-engine prediction at the same flight condition

**Ratios and Deltas (14):** CompressorPR, TurbinePR, CompressorDeltaT, TurbineDeltaT, FuelPerRPM, CorrectedRPM, TempRatioComp, TempRatioTurb, OverallPR, BurnerTempRise, FlowSquared, RPMSquared, FuelFlowRPM, CorrectedFuelFlow

## Split Strategies

| Strategy | Description |
|----------|-------------|
| `official_split` | Holds out a fraction of each engine's own cycles |
| `grouped_split` | Holds out entire engines (harder generalization test) |
