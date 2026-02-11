"""Tests for Preprocessor."""

import numpy as np
import pytest

from mlstudy.core.preprocess import PreprocessConfig, Preprocessor


class TestPreprocessor:
    """Tests for Preprocessor class."""

    def test_fit_uses_train_only_mean(self) -> None:
        """Imputation should use train statistics only."""
        np.random.seed(42)

        # Train has mean ~0
        X_train = np.random.randn(100, 2)
        # Test has mean ~10
        X_test = np.random.randn(50, 2) + 10

        # Add some NaNs
        X_train[0, 0] = np.nan
        X_test[0, 0] = np.nan

        config = PreprocessConfig(impute="mean")
        preprocessor = Preprocessor(config)
        preprocessor.fit(X_train)

        preprocessor.transform(X_train)  # Ensure no error
        X_test_t = preprocessor.transform(X_test)

        # Imputed value in test should be from train mean (~0), not test mean (~10)
        assert abs(X_test_t[0, 0]) < 1  # Should be near 0
        assert X_test_t[0, 0] != X_test[1:, 0].mean()  # Should not be test mean

    def test_fit_uses_train_only_median(self) -> None:
        """Median imputation should use train statistics only."""
        np.random.seed(42)

        X_train = np.array([[1, 2], [3, 4], [5, 6], [np.nan, 8]])
        X_test = np.array([[np.nan, 20], [100, 200]])

        config = PreprocessConfig(impute="median")
        preprocessor = Preprocessor(config)
        preprocessor.fit(X_train)

        X_test_t = preprocessor.transform(X_test)

        # Median of train col 0 is 3 (median of [1, 3, 5])
        assert X_test_t[0, 0] == 3

    def test_fit_uses_train_only_scaling(self) -> None:
        """Scaling should use train statistics only."""
        np.random.seed(42)

        # Train: mean=0, std=1
        X_train = np.random.randn(1000, 2)
        # Test: mean=10, std=2
        X_test = np.random.randn(100, 2) * 2 + 10

        config = PreprocessConfig(scale="standard")
        preprocessor = Preprocessor(config)
        preprocessor.fit(X_train)

        X_test_t = preprocessor.transform(X_test)

        # Test data scaled with train stats should have mean ~10 (not 0)
        assert abs(X_test_t.mean()) > 5  # Far from 0

    def test_transform_before_fit_raises(self) -> None:
        """Should raise error if transform called before fit."""
        config = PreprocessConfig()
        preprocessor = Preprocessor(config)

        with pytest.raises(RuntimeError, match="must be fitted"):
            preprocessor.transform(np.array([[1, 2]]))

    def test_no_imputation(self) -> None:
        """impute='none' should keep NaN values."""
        X = np.array([[1, np.nan], [3, 4]])

        config = PreprocessConfig(impute="none")
        preprocessor = Preprocessor(config)
        X_t = preprocessor.fit_transform(X)

        assert np.isnan(X_t[0, 1])

    def test_no_scaling(self) -> None:
        """scale='none' should keep original values."""
        X = np.array([[10, 20], [30, 40]])

        config = PreprocessConfig(scale="none")
        preprocessor = Preprocessor(config)
        X_t = preprocessor.fit_transform(X)

        np.testing.assert_array_equal(X_t, X)

    def test_standard_scaling(self) -> None:
        """Standard scaling should produce mean~0, std~1 on train."""
        np.random.seed(42)
        X_train = np.random.randn(1000, 2) * 5 + 10  # mean=10, std=5

        config = PreprocessConfig(scale="standard")
        preprocessor = Preprocessor(config)
        X_t = preprocessor.fit_transform(X_train)

        assert abs(X_t.mean()) < 0.1
        assert abs(X_t.std() - 1) < 0.1

    def test_robust_scaling(self) -> None:
        """Robust scaling should use median and IQR."""
        X = np.array([[1], [2], [3], [4], [5], [100]])  # Outlier at 100

        config = PreprocessConfig(scale="robust")
        preprocessor = Preprocessor(config)
        X_t = preprocessor.fit_transform(X)

        # Median should be centered near 0
        assert abs(np.median(X_t)) < 0.5

    def test_winsorization(self) -> None:
        """Winsorization should clip extreme values."""
        X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [100]])

        config = PreprocessConfig(winsorize=0.1)  # 10th and 90th percentile
        preprocessor = Preprocessor(config)
        X_t = preprocessor.fit_transform(X)

        # 100 should be clipped
        assert X_t.max() < 100

    def test_fit_transform_same_as_separate(self) -> None:
        """fit_transform should give same result as fit then transform."""
        np.random.seed(42)
        X = np.random.randn(100, 3)

        config = PreprocessConfig(impute="median", scale="standard")

        preprocessor1 = Preprocessor(config)
        result1 = preprocessor1.fit_transform(X)

        preprocessor2 = Preprocessor(config)
        preprocessor2.fit(X)
        result2 = preprocessor2.transform(X)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_is_fitted_property(self) -> None:
        """is_fitted should reflect fit status."""
        config = PreprocessConfig()
        preprocessor = Preprocessor(config)

        assert not preprocessor.is_fitted

        preprocessor.fit(np.array([[1, 2], [3, 4]]))

        assert preprocessor.is_fitted

    def test_get_params(self) -> None:
        """get_params should return fitted parameters."""
        X = np.array([[1, 2], [3, 4], [5, 6]])

        config = PreprocessConfig(impute="median", scale="standard")
        preprocessor = Preprocessor(config)
        preprocessor.fit(X)

        params = preprocessor.get_params()

        assert "config" in params
        assert "impute_values" in params
        assert "scale_center" in params
        assert "scale_scale" in params
