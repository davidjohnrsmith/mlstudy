"""End-to-end smoke tests for training pipeline."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.core.preprocess import PreprocessConfig
from mlstudy.ml.train.run_experiment import ExperimentConfig, run_experiment


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Create synthetic regression dataset."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    # Generate features and target with some signal
    feature_1 = np.random.randn(n)
    feature_2 = np.random.randn(n)
    target = 0.5 * feature_1 + 0.3 * feature_2 + np.random.randn(n) * 0.5

    return pd.DataFrame(
        {
            "date": dates,
            "feature_1": feature_1,
            "feature_2": feature_2,
            "target": target,
        }
    )


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """Create synthetic classification dataset."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    feature_1 = np.random.randn(n)
    feature_2 = np.random.randn(n)
    # Binary classification based on features
    prob = 1 / (1 + np.exp(-(feature_1 + feature_2)))
    target = (prob > 0.5).astype(int)

    return pd.DataFrame(
        {
            "date": dates,
            "feature_1": feature_1,
            "feature_2": feature_2,
            "target": target,
        }
    )


class TestRegressionTimeSplit:
    """Smoke tests for regression with time split."""

    def test_runs_without_error(self, regression_df: pd.DataFrame, tmp_path) -> None:
        """Should complete without error."""
        config = ExperimentConfig(
            task="regression",
            model_name="ridge",
            split_strategy="time",
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1", "feature_2"],
            preprocess=PreprocessConfig(impute="median", scale="standard"),
            train_end="2024-06-30",
            val_end="2024-09-30",
            test_end="2024-12-31",
        )

        result = run_experiment(regression_df, config, output_dir=str(tmp_path))

        assert result is not None
        assert "mae" in result.test_metrics
        assert "r2" in result.test_metrics
        assert "spearman_ic" in result.test_metrics

    def test_linear_model(self, regression_df: pd.DataFrame, tmp_path) -> None:
        """Should work with linear regression."""
        config = ExperimentConfig(
            task="regression",
            model_name="linear",
            split_strategy="time",
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1", "feature_2"],
            train_end="2024-06-30",
            val_end="2024-09-30",
        )

        result = run_experiment(regression_df, config, output_dir=str(tmp_path))
        assert result.test_metrics["r2"] > -1  # Should have reasonable R2

    def test_rf_model(self, regression_df: pd.DataFrame, tmp_path) -> None:
        """Should work with random forest."""
        config = ExperimentConfig(
            task="regression",
            model_name="rf",
            split_strategy="time",
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1", "feature_2"],
            train_end="2024-06-30",
            val_end="2024-09-30",
        )

        result = run_experiment(regression_df, config, output_dir=str(tmp_path))
        assert "mae" in result.test_metrics

    def test_outputs_saved(self, regression_df: pd.DataFrame, tmp_path) -> None:
        """Should save metrics and predictions."""
        config = ExperimentConfig(
            task="regression",
            model_name="ridge",
            split_strategy="time",
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1", "feature_2"],
            train_end="2024-06-30",
            val_end="2024-09-30",
        )

        run_experiment(regression_df, config, output_dir=str(tmp_path))

        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "predictions.csv").exists()
        assert (tmp_path / "model.joblib").exists()


class TestClassificationWalkForward:
    """Smoke tests for classification with walk-forward."""

    def test_runs_without_error(
        self, classification_df: pd.DataFrame, tmp_path
    ) -> None:
        """Should complete without error."""
        config = ExperimentConfig(
            task="classification",
            model_name="logistic",
            split_strategy="walk_forward",
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1", "feature_2"],
            preprocess=PreprocessConfig(impute="median", scale="standard"),
            train_days=90,
            val_days=30,
            test_days=30,
            step_days=60,
            expanding=True,
        )

        result = run_experiment(classification_df, config, output_dir=str(tmp_path))

        assert result is not None
        assert "accuracy" in result.test_metrics
        assert "f1" in result.test_metrics
        assert len(result.fold_metrics) > 0

    def test_rf_classifier(self, classification_df: pd.DataFrame, tmp_path) -> None:
        """Should work with random forest classifier."""
        config = ExperimentConfig(
            task="classification",
            model_name="rf",
            split_strategy="walk_forward",
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1", "feature_2"],
            train_days=90,
            val_days=30,
            test_days=30,
            step_days=60,
        )

        result = run_experiment(classification_df, config, output_dir=str(tmp_path))
        assert result.test_metrics["accuracy"] > 0

    def test_fold_metrics_present(
        self, classification_df: pd.DataFrame, tmp_path
    ) -> None:
        """Walk-forward should have per-fold metrics."""
        config = ExperimentConfig(
            task="classification",
            model_name="logistic",
            split_strategy="walk_forward",
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1", "feature_2"],
            train_days=90,
            val_days=30,
            test_days=30,
            step_days=60,
        )

        result = run_experiment(classification_df, config, output_dir=str(tmp_path))

        assert len(result.fold_metrics) >= 1
        for fold in result.fold_metrics:
            assert "fold_id" in fold
            assert "train_metrics" in fold
            assert "test_metrics" in fold

    def test_predictions_have_fold_id(
        self, classification_df: pd.DataFrame, tmp_path
    ) -> None:
        """Walk-forward predictions should include fold_id."""
        config = ExperimentConfig(
            task="classification",
            model_name="logistic",
            split_strategy="walk_forward",
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1", "feature_2"],
            train_days=90,
            val_days=30,
            test_days=30,
            step_days=60,
        )

        result = run_experiment(classification_df, config, output_dir=str(tmp_path))

        assert result.predictions is not None
        assert "fold_id" in result.predictions.columns
        assert "y_pred" in result.predictions.columns


class TestPreprocessingIntegration:
    """Test preprocessing within training pipeline."""

    def test_winsorization_applied(
        self, regression_df: pd.DataFrame, tmp_path
    ) -> None:
        """Winsorization should be applied during training."""
        # Add outliers
        df = regression_df.copy()
        df.loc[0, "feature_1"] = 1000

        config = ExperimentConfig(
            task="regression",
            model_name="ridge",
            split_strategy="time",
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1", "feature_2"],
            preprocess=PreprocessConfig(winsorize=0.01),
            train_end="2024-06-30",
            val_end="2024-09-30",
        )

        # Should complete without error despite outlier
        result = run_experiment(df, config, output_dir=str(tmp_path))
        assert result is not None

    def test_imputation_handles_nan(
        self, regression_df: pd.DataFrame, tmp_path
    ) -> None:
        """Imputation should handle NaN values."""
        df = regression_df.copy()
        df.loc[0, "feature_1"] = np.nan
        df.loc[10, "feature_2"] = np.nan

        config = ExperimentConfig(
            task="regression",
            model_name="ridge",
            split_strategy="time",
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1", "feature_2"],
            preprocess=PreprocessConfig(impute="median"),
            train_end="2024-06-30",
            val_end="2024-09-30",
        )

        result = run_experiment(df, config, output_dir=str(tmp_path))
        assert result is not None
