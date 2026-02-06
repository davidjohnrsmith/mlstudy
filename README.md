# mlstudy

ML models for trading signals, alpha generation, and strategy research. Focused on fixed income curve trading with butterfly spreads, mean-reversion signals, and robust backtesting.

## Setup (Windows)

### Step 1: Create Conda Environment

```powershell
conda env create -f environment.yaml
conda activate mlstudy
```

### Step 2: Configure Poetry to Use Conda's Python

```powershell
# IMPORTANT: Run this once after activating the conda env
# This tells Poetry to use the active Python (conda's) instead of creating its own venv
poetry config virtualenvs.create false --local
```

### Step 3: Install Dependencies

```powershell
poetry install
```

## Running Tests

```powershell
poetry run pytest
```

With coverage:

```powershell
poetry run pytest --cov=mlstudy --cov-report=term-missing
```

## Linting & Formatting

```powershell
# Check code style
poetry run ruff check .

# Auto-fix issues
poetry run ruff check --fix .

# Format code
poetry run ruff format .

# Type checking (optional)
poetry run mypy src/
```

## Project Structure

```
mlstudy/
├── src/mlstudy/          # Source code (import as `import mlstudy`)
│   ├── core/             # Foundation: data handling, features, preprocessing
│   ├── ml/               # Machine learning: models, training, uncertainty
│   ├── trading/          # Trading: strategies, backtesting, portfolio
│   ├── deploy/           # Deployment: export, inference, serving
│   └── research/         # Research: simulation, analysis
├── scripts/              # CLI scripts organized by domain
│   ├── data/             # Data simulation scripts
│   ├── ml/               # Training scripts
│   ├── trading/          # Backtest scripts
│   ├── deploy/           # Export scripts
│   └── research/         # Analysis scripts
├── tests/                # Unit and integration tests
├── notebooks/            # Jupyter notebooks for exploration
├── configs/              # YAML configuration files
├── data/                 # Raw and processed data (gitignored)
└── outputs/              # Backtest results, plots, models (gitignored)
```

---

## Module Reference

### `mlstudy.core` — Foundation Layer

Core data handling, feature engineering, and preprocessing utilities.

#### `core.data`
Dataset abstractions and panel data utilities.

| Submodule | Purpose |
|-----------|---------|
| `dataset` | `MLDataFrameDataset` wrapper for time-series data with datetime/target column management |
| `panel` | Panel data validation and reshaping: pivot wide/long, fill gaps, validate structure |
| `session` | Intraday session handling: filter by trading hours, add session flags, timezone support |

#### `core.features`
Feature engineering with registry pattern and composable pipelines.

| Submodule | Purpose |
|-----------|---------|
| `base` | `FeatureSpec`, `FeatureResult`, `FeatureReport` abstractions |
| `pipeline` | `build_features()` to construct features from specs |
| `registry` | Decorator-based feature registration and lookup |
| `time_series/` | Time-series features: returns, volatility, momentum, EWMA |
| `cross_sectional/` | Cross-sectional features: ranking, z-scoring across assets |
| `calendar/` | Calendar features: day of week, month, seasonality encoding |
| `ops/` | Low-level operations: alignment, groupby, pairwise, lagging |

#### `core.preprocess`
Train-only preprocessing to prevent lookahead bias.

| Component | Purpose |
|-----------|---------|
| `PreprocessConfig` | Configuration for imputation, scaling, winsorization |
| `Preprocessor` | Fit on training data only, transform train/test consistently |
| Methods | Mean/median imputation, standard/robust scaling, winsorization |

#### `core.splitters`
Time-aware data splitting for financial time-series.

| Submodule | Purpose |
|-----------|---------|
| `time` | Simple train/val/test splits by date cutoffs |
| `walk_forward` | Walk-forward cross-validation with expanding or rolling windows |

#### `core.utils`
Shared utility functions.

---

### `mlstudy.ml` — Machine Learning Layer

Model training, prediction, and uncertainty quantification.

#### `ml.models`
Model registry for sklearn-compatible estimators.

| Model | Description |
|-------|-------------|
| `linear` | LinearRegression, Ridge |
| `tree` | RandomForestRegressor, HistGradientBoostingRegressor |
| `xgboost` | XGBRegressor (optional dependency) |
| `lightgbm` | LGBMRegressor (optional dependency) |

#### `ml.pipeline`
Dataset construction from raw data.

| Component | Purpose |
|-----------|---------|
| `make_dataset` | Build supervised dataset with features and targets from raw panel data |

#### `ml.targets`
Target variable generation.

| Submodule | Purpose |
|-----------|---------|
| `returns` | Forward returns: `fwd_ret_1d`, `fwd_ret_5d`, etc. |
| `horizons` | Multi-horizon targets for simultaneous prediction |
| `labels` | Directional labels (up/down/flat) for classification |

#### `ml.train`
Experiment management and training utilities.

| Component | Purpose |
|-----------|---------|
| `ExperimentConfig` | Configuration for model, task, hyperparameters |
| `run_experiment` | Run training with fold management and metrics logging |
| `multi_horizon` | Train models for multiple prediction horizons |
| `metrics` | Regression/classification metrics: RMSE, MAE, R², accuracy, F1 |

#### `ml.uncertainty`
Prediction intervals and calibration.

| Component | Purpose |
|-----------|---------|
| `QuantilePredictor` | Multi-quantile regression via LightGBM |
| `conformal` | Conformal prediction for calibrated coverage guarantees |

---

### `mlstudy.trading` — Trading Layer

Strategies, signals, and backtesting infrastructure.

#### `trading.strategy`
Trading strategy implementations.

| Submodule | Purpose |
|-----------|---------|
| `fly/fly` | Butterfly spread construction: compute fly values, DV01-neutral weights |
| `fly/fly_universe` | Generate all valid flies from tenors, parameter sweep utilities |
| `fly/leg_selection` | Select fly legs by tenor, daily leg selection for intraday stability |
| `fly/curve_selection` | Curve-based leg selection logic |
| `mean_reversion/signals` | Z-score signals: rolling/EWMA, entry/exit/stop thresholds, position tracking |
| `momentum/` | Momentum-based signal generation |
| `regime/` | Regime detection and strategy adaptation |

#### `trading.backtest`
Backtesting engines and performance analysis.

| Component | Purpose |
|-----------|---------|
| `engine` | Daily backtest: `backtest_fly()`, `backtest_fly_from_panel()` |
| `intraday` | Intraday backtest: session-aware execution, rebalance modes (open_only, every_bar) |
| `BacktestConfig` | Position sizing: `SizingMode.FIXED_NOTIONAL`, `SizingMode.DV01_TARGET` |
| `metrics` | `BacktestMetrics`: Sharpe, Sortino, Calmar, drawdown, turnover, hit rate, VaR/CVaR |
| `report` | Generate plots (cumulative P&L, drawdown, returns histogram), print summaries |

#### `trading.portfolio`
Portfolio-level backtesting and aggregation.

| Component | Purpose |
|-----------|---------|
| `backtest` | Multi-strategy portfolio backtesting with DV01 targeting |
| `weighting` | Position weighting schemes |
| `aggregate` | Aggregate multiple strategy results |
| `signal_adapter` | Adapt signals to portfolio framework |

---

### `mlstudy.deploy` — Deployment Layer

Model export, inference, and serving for production.

#### `deploy.export`
Bundle models for deployment.

| Component | Purpose |
|-----------|---------|
| `artifacts` | Package trained model + preprocessing + metadata into deployable artifact |

#### `deploy.inference`
Portable model inference without sklearn dependencies.

| Component | Purpose |
|-----------|---------|
| `base` | Base inference interface |
| `linear` | Pure numpy inference for linear models |
| `xgboost_inf` | XGBoost booster serialization and inference |
| `lightgbm_inf` | LightGBM booster serialization and inference |
| `reconstruct` | Reconstruct model from exported artifacts |
| `export` | Export model weights to JSON + numpy arrays |

#### `deploy.serve`
Model serving utilities.

| Component | Purpose |
|-----------|---------|
| `ArtifactPredictor` | Load exported artifacts and run inference |

---

### `mlstudy.research` — Research Layer

Simulation and analysis utilities for strategy research.

#### `research.simulate`
Synthetic data generation for testing.

| Component | Purpose |
|-----------|---------|
| `market` | Generate synthetic panel market data with configurable: asset count, date range, regime shifts, flow signals |

#### `research.analysis`
Statistical analysis utilities.

| Submodule | Purpose |
|-----------|---------|
| `distributions/grouped` | Group-wise distribution analysis |
| `distributions/metrics` | Statistical comparison metrics |
| `distributions/plots` | Distribution visualization |
| `distributions/report` | Generate distribution analysis reports |

---

## CLI Scripts

Scripts are organized by domain under `scripts/`.

### Trading Scripts (`scripts/trading/`)

| Script | Purpose |
|--------|---------|
| `run_fly_backtest.py` | Daily fly backtest with z-score signals |
| `run_fly_backtest_intraday.py` | Intraday fly backtest with session-aware execution |
| `run_fly_backtest_momentum.py` | Momentum-based fly backtest |
| `run_portfolio_backtest.py` | Portfolio-level backtest |
| `sweep_fly_backtests.py` | Parameter sweep across flies and signal parameters |

### ML Scripts (`scripts/ml/`)

| Script | Purpose |
|--------|---------|
| `train.py` | Train ML models with time or walk-forward splits |
| `train_multi_horizon.py` | Train multi-horizon prediction models |
| `train_on_prepared.py` | Train on pre-built supervised dataset |

### Data Scripts (`scripts/data/`)

| Script | Purpose |
|--------|---------|
| `simulate_data.py` | Generate synthetic market data |

### Deploy Scripts (`scripts/deploy/`)

| Script | Purpose |
|--------|---------|
| `export_artifact.py` | Export trained model as deployment artifact |

### Research Scripts (`scripts/research/`)

| Script | Purpose |
|--------|---------|
| `compare_group_distributions.py` | Statistical comparison of feature distributions |

## Example Usage

### Fly Backtest (Daily)

```python
from mlstudy.trading.backtest import backtest_fly_from_panel, BacktestConfig, SizingMode
from mlstudy.trading.strategy.structures.specs.fly import select_fly_legs

# Select 2y5y10y fly legs from panel
legs = select_fly_legs(panel_df, tenors=(2, 5, 10))

# Run backtest with DV01 target sizing
config = BacktestConfig(sizing_mode=SizingMode.DV01_TARGET, dv01_target=10000)
result = backtest_fly_from_panel(panel_df, legs, window=20, entry_z=2.0, config=config)

print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

### Fly Backtest (Intraday)

```python
from mlstudy.trading.backtest import backtest_fly_intraday, IntradayBacktestConfig

config = IntradayBacktestConfig(
    session_start="07:30",
    session_end="17:00",
    tz="Europe/Berlin",
    rebalance_mode="open_only",  # Only trade at session open
    sizing_mode=SizingMode.DV01_TARGET,
)

result = backtest_fly_intraday(
    panel_df,
    tenors=(2, 5, 10),
    window=20,
    entry_z=2.0,
    config=config,
)
```

### Parameter Sweep

```python
from mlstudy.trading.strategy.structures.specs.fly import generate_flies_from_tenors, build_and_backtest_many_flies
from mlstudy.trading.backtest import ParamGrid

# Generate all valid flies from tenors
flies = generate_flies_from_tenors([2, 3, 5, 7, 10, 30])

# Define parameter grid
param_grid = ParamGrid(
    windows=[10, 20, 30],
    entry_zs=[1.5, 2.0, 2.5],
    exit_zs=[0.3, 0.5],
)

# Run sweep
results_df = build_and_backtest_many_flies(panel_df, flies, param_grid)

# Find best by Sharpe
best = results_df.loc[results_df["sharpe_ratio"].idxmax()]
```

### Feature Engineering

```python
from mlstudy.core.features import build_features, FeatureSpec

specs = [
    FeatureSpec(name="ret_1d", kind="return", params={"periods": 1}),
    FeatureSpec(name="vol_20d", kind="volatility", params={"window": 20}),
    FeatureSpec(name="rank_ret", kind="cross_sectional_rank", params={"column": "ret_1d"}),
]

df, report = build_features(df, specs, datetime_col="datetime", group_col="asset_id")
```

### Training with Walk-Forward

```python
from mlstudy.ml.train import run_experiment, ExperimentConfig
from mlstudy.core.splitters import walk_forward_splits
from mlstudy.core.data import MLDataFrameDataset

dataset = MLDataFrameDataset(df, datetime_col="datetime", target_col="fwd_ret")
folds = walk_forward_splits(dataset, n_folds=5, train_months=12, test_months=1)

config = ExperimentConfig(model_name="ridge", task="regression")
result = run_experiment(dataset, folds, config)
```

### Model Export & Inference

```python
from mlstudy.deploy.export import export_artifact
from mlstudy.deploy.serve import ArtifactPredictor

# Export trained model
export_artifact(model, preprocessor, metadata, output_dir="artifacts/model_v1")

# Load and serve predictions (no sklearn dependency needed)
predictor = ArtifactPredictor.load("artifacts/model_v1")
predictions = predictor.predict(new_data)
```

### Synthetic Data Generation

```python
from mlstudy.research.simulate import generate_market_data

# Generate synthetic panel data for testing
df = generate_market_data(
    n_assets=10,
    n_days=252,
    with_regime_shifts=True,
    with_flow_signals=True,
)
```

## Conventions

### Data Handling

- **Never commit data files** to version control
- Place raw data in `data/raw/`, processed data in `data/processed/`
- Use configuration files to specify data paths
- Document data sources and preprocessing steps

### Outputs & Artifacts

- Backtest results, plots, and trained models go in `outputs/`
- Use timestamped subdirectories: `outputs/2024-01-15_experiment_name/`
- The `outputs/` directory is gitignored; archive important results elsewhere

### Reproducibility

- Pin dependencies only when necessary (Python 3.7+ compatible)
- Log random seeds and hyperparameters in experiment configs
- Use `configs/` for YAML-based experiment configuration

### Credentials & API Keys

- **Never commit credentials** (broker APIs, data vendor keys)
- Use environment variables or a `.env` file (gitignored)
- See `.env.example` for required variables template
