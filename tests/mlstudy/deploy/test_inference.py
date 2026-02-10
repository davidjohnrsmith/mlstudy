"""Tests for inference module."""

import numpy as np
import pytest

from mlstudy.deploy.inference.linear import (
    LinearInferenceModel,
)
from mlstudy.deploy.inference.export import (
    export_model,
    get_model_type,
)
from mlstudy.deploy.inference.reconstruct import (

    load_inference_model,
)
@pytest.fixture
def regression_data():
    """Simple regression data."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1, 2, 3, 4, 5]) + 0.5 + np.random.randn(100) * 0.1
    return X, y


@pytest.fixture
def classification_data():
    """Simple binary classification data."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


class TestLinearInferenceModel:
    """Tests for LinearInferenceModel."""

    def test_regression_predict(self, regression_data):
        """Should predict correctly for regression."""
        X, y = regression_data

        from sklearn.linear_model import Ridge

        model = Ridge(alpha=0.1).fit(X, y)

        # Create inference model
        inf_model = LinearInferenceModel.from_sklearn(model, task="regression")

        # Compare predictions
        sklearn_preds = model.predict(X)
        inf_preds = inf_model.predict(X)

        np.testing.assert_allclose(sklearn_preds, inf_preds, rtol=1e-10)

    def test_classification_predict(self, classification_data):
        """Should predict correctly for classification."""
        X, y = classification_data

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression().fit(X, y)

        # Create inference model
        inf_model = LinearInferenceModel.from_sklearn(model, task="classification")

        # Compare predictions
        sklearn_preds = model.predict(X)
        inf_preds = inf_model.predict(X)

        np.testing.assert_array_equal(sklearn_preds, inf_preds)

    def test_classification_predict_proba(self, classification_data):
        """Should predict probabilities correctly."""
        X, y = classification_data

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression().fit(X, y)
        inf_model = LinearInferenceModel.from_sklearn(model, task="classification")

        sklearn_proba = model.predict_proba(X)
        inf_proba = inf_model.predict_proba(X)

        np.testing.assert_allclose(sklearn_proba, inf_proba, rtol=1e-10)

    def test_save_load_regression(self, regression_data, tmp_path):
        """Should save and load correctly."""
        X, y = regression_data

        from sklearn.linear_model import Ridge

        model = Ridge(alpha=0.1).fit(X, y)
        inf_model = LinearInferenceModel.from_sklearn(model, task="regression")

        # Save
        save_path = tmp_path / "model"
        inf_model.save(save_path)

        assert (save_path / "coef.npy").exists()
        assert (save_path / "intercept.npy").exists()
        assert (save_path / "metadata.json").exists()

        # Load
        loaded_model = LinearInferenceModel.load(save_path)

        # Compare predictions
        orig_preds = inf_model.predict(X)
        loaded_preds = loaded_model.predict(X)

        np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-10)

    def test_save_load_classification(self, classification_data, tmp_path):
        """Should save and load classification model correctly."""
        X, y = classification_data

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression().fit(X, y)
        inf_model = LinearInferenceModel.from_sklearn(model, task="classification")

        # Save and load
        save_path = tmp_path / "model"
        inf_model.save(save_path)
        loaded_model = LinearInferenceModel.load(save_path)

        # Compare predictions
        np.testing.assert_array_equal(
            inf_model.predict(X), loaded_model.predict(X)
        )


class TestExportModel:
    """Tests for export_model function."""

    def test_export_linear_regression(self, regression_data, tmp_path):
        """Should export linear regression model."""
        X, y = regression_data

        from sklearn.linear_model import LinearRegression

        model = LinearRegression().fit(X, y)

        # Export
        artifacts = export_model(model, tmp_path / "model", task="regression")

        assert "coef" in artifacts
        assert "metadata" in artifacts

    def test_export_logistic_regression(self, classification_data, tmp_path):
        """Should export logistic regression model."""
        X, y = classification_data

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression().fit(X, y)

        artifacts = export_model(model, tmp_path / "model", task="classification")

        assert "coef" in artifacts

    def test_get_model_type_linear(self):
        """Should detect linear model types."""
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

        assert get_model_type(LinearRegression()) == "linear"
        assert get_model_type(Ridge()) == "linear"
        assert get_model_type(LogisticRegression()) == "linear"


class TestLoadInferenceModel:
    """Tests for load_inference_model function."""

    def test_load_linear(self, regression_data, tmp_path):
        """Should load linear model correctly."""
        X, y = regression_data

        from sklearn.linear_model import Ridge

        model = Ridge().fit(X, y)
        export_model(model, tmp_path / "model", task="regression")

        # Load using generic function
        loaded = load_inference_model(tmp_path / "model")

        assert isinstance(loaded, LinearInferenceModel)
        assert loaded.task == "regression"

    def test_load_preserves_predictions(self, regression_data, tmp_path):
        """Loaded model should produce same predictions."""
        X, y = regression_data

        from sklearn.linear_model import Ridge

        model = Ridge().fit(X, y)
        original_preds = model.predict(X)

        export_model(model, tmp_path / "model", task="regression")
        loaded = load_inference_model(tmp_path / "model")

        loaded_preds = loaded.predict(X)

        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-10)


class TestXGBoostInference:
    """Tests for XGBoost inference (if xgboost available)."""

    @pytest.fixture
    def xgb_available(self):
        """Check if xgboost is available."""
        import importlib.util

        if importlib.util.find_spec("xgboost") is None:
            pytest.skip("xgboost not installed")
        return True

    def test_export_xgboost_json(self, regression_data, tmp_path, xgb_available):
        """Should export XGBoost model to JSON."""
        import xgboost as xgb

        X, y = regression_data
        model = xgb.XGBRegressor(n_estimators=10, max_depth=3).fit(X, y)

        export_model(
            model, tmp_path / "model", task="regression", save_format="json"
        )

        assert (tmp_path / "model" / "model.json").exists()

    def test_xgboost_roundtrip(self, regression_data, tmp_path, xgb_available):
        """Should save and load XGBoost model correctly."""
        import xgboost as xgb

        X, y = regression_data
        model = xgb.XGBRegressor(n_estimators=10, max_depth=3).fit(X, y)
        original_preds = model.predict(X)

        export_model(model, tmp_path / "model", task="regression")
        loaded = load_inference_model(tmp_path / "model")

        loaded_preds = loaded.predict(X)

        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)


class TestLightGBMInference:
    """Tests for LightGBM inference (if lightgbm available)."""

    @pytest.fixture
    def lgb_available(self):
        """Check if lightgbm is available."""
        import importlib.util

        if importlib.util.find_spec("lightgbm") is None:
            pytest.skip("lightgbm not installed")
        return True

    def test_export_lightgbm(self, regression_data, tmp_path, lgb_available):
        """Should export LightGBM model to text file."""
        import lightgbm as lgb

        X, y = regression_data
        model = lgb.LGBMRegressor(n_estimators=10, max_depth=3, verbose=-1).fit(X, y)

        export_model(model, tmp_path / "model", task="regression")

        assert (tmp_path / "model" / "model.txt").exists()

    def test_lightgbm_roundtrip(self, regression_data, tmp_path, lgb_available):
        """Should save and load LightGBM model correctly."""
        import lightgbm as lgb

        X, y = regression_data
        model = lgb.LGBMRegressor(n_estimators=10, max_depth=3, verbose=-1).fit(X, y)
        original_preds = model.predict(X)

        export_model(model, tmp_path / "model", task="regression")
        loaded = load_inference_model(tmp_path / "model")

        loaded_preds = loaded.predict(X)

        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)
