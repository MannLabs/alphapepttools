import anndata as ad
import numpy as np
import pytest

from alphatools.pp import normalize
from alphatools.pp.norm import _total_mean_normalization, _total_median_normalization, _validate_strategies


@pytest.fixture
def all_equal_count_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate count data (samples, features) where all samples have the same intensity,
    the expected result and the expected normalization factors"""
    X = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])

    assert X.sum(axis=1).all()

    return (
        X,
        X,
        np.array([1.0, 1.0, 1.0]),
    )


@pytest.fixture
def different_count_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate count data (samples, features) where samples have different intensities,
    the expected result and the expected normalization factors"""
    # Reduce total intensity of sample 0
    return (
        np.array([[0.8, 1.0], [2.0, 0.0], [0.0, 2.0]]),
        {
            "total_mean": np.array([[0.85925926, 1.07407407], [1.93333333, 0.0], [0.0, 1.93333333]]),
            "total_median": np.array([[0.88888889, 1.11111111], [2.0, 0.0], [0.0, 2.0]]),
        },
        {
            "total_mean": np.array([1.07407407, 0.96666667, 0.96666667]),
            "total_median": np.array([1.111111, 1.0, 1.0]),
        },
    )


@pytest.fixture
def nan_count_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate count data (samples, features) where samples have different intensities and contain nan values,
    the expected result and the expected normalization factors"""
    # Add nan intensites
    return (
        np.array([[0, 1.0], [2.0, 0.0], [0.8, np.nan], [np.nan, 2.0]]),
        {
            "total_mean": np.array([[0, 1.45], [1.45, 0.0], [1.45, np.nan], [np.nan, 1.45]]),
            "total_median": np.array([[0.0, 1.5], [1.5, 0.0], [1.5, np.nan], [np.nan, 1.5]]),
        },
        {"total_mean": np.array([1.45, 0.725, 1.8124999, 0.725]), "total_median": np.array([1.5, 0.75, 1.875, 0.75])},
    )


class TestValidation:
    """Test input validation"""

    @pytest.mark.parametrize("strategy", ["total_mean", "total_median"])
    def test__validate_strategies(self, strategy) -> None:
        # Valid strategy
        _validate_strategies(strategy)

    def test__valid_strategies_invalid(self) -> None:
        # Invalid strategy
        with pytest.raises(ValueError, match="`strategy` must be one of"):
            _validate_strategies("invalid_strategy")


class TestNormalizationFunctions:
    """Test low-level normalization fucntions"""

    def test__total_mean_normalization_all_equal(self, all_equal_count_data) -> None:
        """Test that equal sample intensities lead to no change in data values for total mean normalization"""
        array, norm_array_ref, norm_factors_ref = all_equal_count_data
        norm_array, norm_factors = _total_mean_normalization(array)

        assert np.isclose(norm_array, norm_array_ref, atol=1e-6).all()
        assert np.isclose(norm_factors, norm_factors_ref, atol=1e-6).all()

    def test__total_median_normalization_all_equal(self, all_equal_count_data) -> None:
        """Test that equal sample intensities lead to identity transform for total median normalization"""

        array, norm_array_ref, norm_factors_ref = all_equal_count_data
        norm_array, norm_factors = _total_median_normalization(array)

        assert np.isclose(norm_array, norm_array_ref, atol=1e-6).all()
        assert np.isclose(norm_factors, norm_factors_ref, atol=1e-6).all()

    def test__mean_normalization_different(self, different_count_data) -> None:
        """Test total mean normalization"""
        STRATEGY = "total_mean"

        array, norm_array_ref, norm_factors_ref = different_count_data
        norm_array, norm_factors = _total_mean_normalization(array)

        assert np.isclose(norm_array, norm_array_ref[STRATEGY], atol=1e-6).all()
        assert np.isclose(norm_factors, norm_factors_ref[STRATEGY], atol=1e-6).all()

    def test__median_normalization_different(self, different_count_data) -> None:
        """Test total median normalization"""
        STRATEGY = "total_median"

        array, norm_array_ref, norm_factors_ref = different_count_data
        norm_array, norm_factors = _total_median_normalization(array)

        assert np.isclose(norm_array, norm_array_ref[STRATEGY], atol=1e-6).all()
        assert np.isclose(norm_factors, norm_factors_ref[STRATEGY], atol=1e-6).all()

    def test__mean_normalization_nan_values(self, nan_count_data) -> None:
        STRATEGY = "total_mean"
        array, norm_array_ref, norm_factors_ref = nan_count_data
        norm_array, norm_factors = _total_mean_normalization(array)

        assert np.isclose(norm_array, norm_array_ref[STRATEGY], atol=1e-6, equal_nan=True).all()
        assert np.isclose(norm_factors, norm_factors_ref[STRATEGY], atol=1e-6, equal_nan=True).all()

    def test__median_normalization_nan_values(self, nan_count_data) -> None:
        STRATEGY = "total_median"
        array, norm_array_ref, norm_factors_ref = nan_count_data
        norm_array, norm_factors = _total_median_normalization(array)

        assert np.isclose(norm_array, norm_array_ref[STRATEGY], atol=1e-6, equal_nan=True).all()
        assert np.isclose(norm_factors, norm_factors_ref[STRATEGY], atol=1e-6, equal_nan=True).all()


class TestNormalizeFunction:
    """Test the high-level normalize function"""

    @pytest.mark.parametrize("strategy", ["total_mean", "total_median"])
    def test_normalize_default_parameters(self, different_count_data, strategy: str) -> None:
        """Test normalize with default parameters (normalizes adata.X in place)"""
        # Create test data
        array, norm_array_ref, _ = different_count_data
        adata = ad.AnnData(X=array.copy())

        # Normalize
        normalize(adata, strategy=strategy)

        # Check that X was normalized
        assert np.isclose(adata.X, norm_array_ref[strategy], atol=1e-6).all()
        assert len(adata.obs.columns) == 0
        assert len(adata.layers) == 0

    @pytest.mark.parametrize("strategy", ["total_mean", "total_median"])
    def test_normalize_key_added(self, different_count_data, strategy: str) -> None:
        """Test normalize with key_added parameter"""
        # Create test data
        array, norm_array_ref, norm_factors_ref = different_count_data
        adata = ad.AnnData(X=array.copy())

        # Normalize
        normalize(adata, strategy=strategy, key_added="norm_factors")

        # Check that X was normalized
        assert np.isclose(adata.X, norm_array_ref[strategy], atol=1e-6).all()
        assert "norm_factors" in adata.obs.columns
        assert np.isclose(adata.obs["norm_factors"].to_numpy(), norm_factors_ref[strategy], atol=1e-6).all()
        assert len(adata.layers) == 0
