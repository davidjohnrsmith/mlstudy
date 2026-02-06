"""Smoke tests for train_on_prepared script."""

import json
from pathlib import Path

import pytest

from mlstudy.core.features.base import FeatureSpec
from mlstudy.ml.pipeline.make_dataset import TargetSpec, make_supervised_dataset
from mlstudy.research.simulate.market import simulate_market_data


@pytest.fixture
def prepared_data_dir(tmp_path) -> Path:
    """Create a small prepared dataset for testing."""
    # Generate small simulated data
    df = simulate_market_data(n_assets=2, n_periods=200, seed=42)

    # Build features
    feature_specs = [
        FeatureSpec(name="returns", params={"price_col": "close"}),
        FeatureSpec(name="momentum", params={"price_col": "close", "window": 5}),
    ]
    target_spec = TargetSpec(horizon_steps=1)

    X, y, meta, _ = make_supervised_dataset(
        df,
        feature_specs,
        target_spec,
        datetime_col="datetime",
        group_col="asset",
        dropna=True,
    )

    # Save to tmp_path
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()

    X.to_csv(data_dir / "X.csv", index=False)
    y.to_csv(data_dir / "y.csv", index=False)
    meta.to_csv(data_dir / "meta.csv", index=False)

    return data_dir


class TestTrainOnPrepared:
    """Smoke tests for training pipeline."""

    def test_regression_time_split(self, prepared_data_dir: Path, tmp_path) -> None:
        """Should complete regression with time split."""
        from scripts.ml.train_on_prepared import main

        output_dir = tmp_path / "output"

        result = main(
            [
                "--data-dir",
                str(prepared_data_dir),
                "--task",
                "regression",
                "--model",
                "ridge",
                "--split",
                "time",
                "--outdir",
                str(output_dir),
            ]
        )

        assert result == 0
        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "predictions.csv").exists()

        # Check metrics.json content
        with open(output_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert "train" in metrics
        assert "val" in metrics
        assert "test" in metrics

    def test_regression_rf_model(self, prepared_data_dir: Path, tmp_path) -> None:
        """Should work with random forest."""
        from scripts.ml.train_on_prepared import main

        output_dir = tmp_path / "output_rf"

        result = main(
            [
                "--data-dir",
                str(prepared_data_dir),
                "--task",
                "regression",
                "--model",
                "rf",
                "--split",
                "time",
                "--outdir",
                str(output_dir),
            ]
        )

        assert result == 0
        assert (output_dir / "metrics.json").exists()

    def test_classification_time_split(self, prepared_data_dir: Path, tmp_path) -> None:
        """Should complete classification with time split."""
        # Create classification target
        df = simulate_market_data(n_assets=2, n_periods=200, seed=42)

        feature_specs = [
            FeatureSpec(name="returns", params={"price_col": "close"}),
        ]
        target_spec = TargetSpec(target_type="forward_direction", horizon_steps=1)

        X, y, meta, _ = make_supervised_dataset(
            df,
            feature_specs,
            target_spec,
            datetime_col="datetime",
            group_col="asset",
            dropna=True,
        )

        # Save classification data
        class_dir = tmp_path / "classification_data"
        class_dir.mkdir()
        X.to_csv(class_dir / "X.csv", index=False)
        y.to_csv(class_dir / "y.csv", index=False)
        meta.to_csv(class_dir / "meta.csv", index=False)

        from scripts.ml.train_on_prepared import main

        output_dir = tmp_path / "output_class"

        result = main(
            [
                "--data-dir",
                str(class_dir),
                "--task",
                "classification",
                "--model",
                "logistic",
                "--split",
                "time",
                "--outdir",
                str(output_dir),
            ]
        )

        assert result == 0
        assert (output_dir / "metrics.json").exists()

        with open(output_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert "accuracy" in metrics["test"]

    def test_walk_forward_split(self, prepared_data_dir: Path, tmp_path) -> None:
        """Should work with walk-forward split."""
        from scripts.ml.train_on_prepared import main

        output_dir = tmp_path / "output_wf"

        result = main(
            [
                "--data-dir",
                str(prepared_data_dir),
                "--task",
                "regression",
                "--model",
                "ridge",
                "--split",
                "walk_forward",
                "--train-days",
                "10",
                "--val-days",
                "3",
                "--test-days",
                "3",
                "--step-days",
                "5",
                "--outdir",
                str(output_dir),
            ]
        )

        assert result == 0
        assert (output_dir / "metrics.json").exists()

        with open(output_dir / "metrics.json") as f:
            metrics = json.load(f)
        # Walk-forward should have fold metrics
        assert "fold_metrics" in metrics

    def test_output_files_exist(self, prepared_data_dir: Path, tmp_path) -> None:
        """Should create all expected output files."""
        from scripts.ml.train_on_prepared import main

        output_dir = tmp_path / "output_full"

        main(
            [
                "--data-dir",
                str(prepared_data_dir),
                "--task",
                "regression",
                "--model",
                "ridge",
                "--split",
                "time",
                "--outdir",
                str(output_dir),
            ]
        )

        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "predictions.csv").exists()
        assert (output_dir / "model.joblib").exists()
        assert (output_dir / "preprocessor.joblib").exists()
