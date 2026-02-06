"""Tests for artifact export and serving."""

import numpy as np
import pytest

from mlstudy.deploy.export.artifacts import ArtifactExporter, export_artifact, load_artifact_manifest
from mlstudy.deploy.serve.predictor import ArtifactPredictor

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


@pytest.fixture
def feature_names():
    """Feature names for test data."""
    return ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]


class TestArtifactExporter:
    """Tests for ArtifactExporter."""

    def test_export_linear_regression(self, regression_data, feature_names, tmp_path):
        """Should export linear regression model."""
        X, y = regression_data

        from sklearn.linear_model import Ridge

        model = Ridge(alpha=0.1).fit(X, y)

        exporter = ArtifactExporter(
            model=model,
            feature_names=feature_names,
            task="regression",
        )
        exporter.export(tmp_path / "artifact")

        # Check files exist
        assert (tmp_path / "artifact" / "manifest.json").exists()
        assert (tmp_path / "artifact" / "feature_schema.json").exists()
        assert (tmp_path / "artifact" / "model" / "linear.json").exists()
        assert (tmp_path / "artifact" / "code_version.txt").exists()

        # Check manifest
        manifest = load_artifact_manifest(tmp_path / "artifact")
        assert manifest["model_type"] == "linear"
        assert manifest["task"] == "regression"
        assert manifest["n_features"] == 5

    def test_export_logistic_regression(self, classification_data, feature_names, tmp_path):
        """Should export logistic regression model."""
        X, y = classification_data

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression().fit(X, y)

        export_artifact(
            model=model,
            path=tmp_path / "artifact",
            feature_names=feature_names,
            task="classification",
        )

        assert (tmp_path / "artifact" / "model" / "linear.json").exists()

        manifest = load_artifact_manifest(tmp_path / "artifact")
        assert manifest["model_type"] == "linear"
        assert manifest["task"] == "classification"

    def test_export_with_metadata(self, regression_data, tmp_path):
        """Should include custom metadata in manifest."""
        X, y = regression_data

        from sklearn.linear_model import Ridge

        model = Ridge().fit(X, y)

        export_artifact(
            model=model,
            path=tmp_path / "artifact",
            task="regression",
            metadata={"experiment": "test_001", "author": "test"},
        )

        manifest = load_artifact_manifest(tmp_path / "artifact")
        assert manifest["experiment"] == "test_001"
        assert manifest["author"] == "test"

    def test_detect_model_type(self, regression_data):
        """Should auto-detect model types."""
        X, y = regression_data

        from sklearn.linear_model import LinearRegression, Ridge

        # Linear
        exporter = ArtifactExporter(model=Ridge().fit(X, y), task="regression")
        assert exporter.model_type == "linear"

        exporter = ArtifactExporter(model=LinearRegression().fit(X, y), task="regression")
        assert exporter.model_type == "linear"


class TestArtifactPredictor:
    """Tests for ArtifactPredictor."""

    def test_load_and_predict_linear_regression(self, regression_data, feature_names, tmp_path):
        """Should load and predict with linear regression."""
        X, y = regression_data

        from sklearn.linear_model import Ridge

        model = Ridge(alpha=0.1).fit(X, y)
        sklearn_preds = model.predict(X)

        # Export
        export_artifact(
            model=model,
            path=tmp_path / "artifact",
            feature_names=feature_names,
            task="regression",
        )

        # Load and predict
        predictor = ArtifactPredictor.load(tmp_path / "artifact")
        artifact_preds = predictor.predict(X)

        # Should match sklearn predictions
        np.testing.assert_allclose(sklearn_preds, artifact_preds, rtol=1e-10)

    def test_load_and_predict_logistic_regression(self, classification_data, feature_names, tmp_path):
        """Should load and predict with logistic regression."""
        X, y = classification_data

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression().fit(X, y)
        sklearn_preds = model.predict(X)
        sklearn_proba = model.predict_proba(X)

        # Export
        export_artifact(
            model=model,
            path=tmp_path / "artifact",
            feature_names=feature_names,
            task="classification",
        )

        # Load and predict
        predictor = ArtifactPredictor.load(tmp_path / "artifact")
        artifact_preds = predictor.predict(X)
        artifact_proba = predictor.predict_proba(X)

        # Should match sklearn predictions
        np.testing.assert_array_equal(sklearn_preds, artifact_preds)
        np.testing.assert_allclose(sklearn_proba, artifact_proba, rtol=1e-10)

    def test_predictor_metadata(self, regression_data, feature_names, tmp_path):
        """Should expose metadata."""
        X, y = regression_data

        from sklearn.linear_model import Ridge

        model = Ridge().fit(X, y)

        export_artifact(
            model=model,
            path=tmp_path / "artifact",
            feature_names=feature_names,
            task="regression",
        )

        predictor = ArtifactPredictor.load(tmp_path / "artifact")

        assert predictor.model_type == "linear"
        assert predictor.task == "regression"
        assert predictor.n_features == 5
        assert predictor.feature_names == feature_names


class TestExportLoadParity:
    """Tests for export -> load -> predict parity."""

    def test_linear_regression_parity(self, regression_data, tmp_path):
        """Linear regression should have exact parity."""
        X, y = regression_data

        from sklearn.linear_model import Ridge

        model = Ridge(alpha=0.1).fit(X, y)
        original_preds = model.predict(X)

        export_artifact(model=model, path=tmp_path / "artifact", task="regression")
        predictor = ArtifactPredictor.load(tmp_path / "artifact")
        loaded_preds = predictor.predict(X)

        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-10)

    def test_logistic_regression_parity(self, classification_data, tmp_path):
        """Logistic regression should have exact parity."""
        X, y = classification_data

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression().fit(X, y)
        original_preds = model.predict(X)
        original_proba = model.predict_proba(X)

        export_artifact(model=model, path=tmp_path / "artifact", task="classification")
        predictor = ArtifactPredictor.load(tmp_path / "artifact")
        loaded_preds = predictor.predict(X)
        loaded_proba = predictor.predict_proba(X)

        np.testing.assert_array_equal(original_preds, loaded_preds)
        np.testing.assert_allclose(original_proba, loaded_proba, rtol=1e-10)


class TestXGBoostArtifacts:
    """Tests for XGBoost artifact export (if available)."""

    @pytest.fixture
    def xgb_available(self):
        """Check if xgboost is available."""
        import importlib.util

        if importlib.util.find_spec("xgboost") is None:
            pytest.skip("xgboost not installed")
        return True

    def test_xgboost_export_load_parity(self, regression_data, tmp_path, xgb_available):
        """XGBoost should have prediction parity."""
        import xgboost as xgb

        X, y = regression_data
        model = xgb.XGBRegressor(n_estimators=10, max_depth=3).fit(X, y)
        original_preds = model.predict(X)

        export_artifact(model=model, path=tmp_path / "artifact", task="regression")
        predictor = ArtifactPredictor.load(tmp_path / "artifact")
        loaded_preds = predictor.predict(X)

        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)

    def test_xgboost_classification_parity(self, classification_data, tmp_path, xgb_available):
        """XGBoost classification should have prediction parity."""
        import xgboost as xgb

        X, y = classification_data
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, eval_metric="logloss").fit(X, y)
        original_preds = model.predict(X)

        export_artifact(model=model, path=tmp_path / "artifact", task="classification")
        predictor = ArtifactPredictor.load(tmp_path / "artifact")
        loaded_preds = predictor.predict(X)

        np.testing.assert_array_equal(original_preds, loaded_preds)


class TestLightGBMArtifacts:
    """Tests for LightGBM artifact export (if available)."""

    @pytest.fixture
    def lgb_available(self):
        """Check if lightgbm is available."""
        import importlib.util

        if importlib.util.find_spec("lightgbm") is None:
            pytest.skip("lightgbm not installed")
        return True

    def test_lightgbm_export_load_parity(self, regression_data, tmp_path, lgb_available):
        """LightGBM should have prediction parity."""
        import lightgbm as lgb

        X, y = regression_data
        model = lgb.LGBMRegressor(n_estimators=10, max_depth=3, verbose=-1).fit(X, y)
        original_preds = model.predict(X)

        export_artifact(model=model, path=tmp_path / "artifact", task="regression")
        predictor = ArtifactPredictor.load(tmp_path / "artifact")
        loaded_preds = predictor.predict(X)

        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)

    def test_lightgbm_classification_parity(self, classification_data, tmp_path, lgb_available):
        """LightGBM classification should have prediction parity."""
        import lightgbm as lgb

        X, y = classification_data
        model = lgb.LGBMClassifier(n_estimators=10, max_depth=3, verbose=-1).fit(X, y)
        original_preds = model.predict(X)

        export_artifact(model=model, path=tmp_path / "artifact", task="classification")
        predictor = ArtifactPredictor.load(tmp_path / "artifact")
        loaded_preds = predictor.predict(X)

        np.testing.assert_array_equal(original_preds, loaded_preds)
