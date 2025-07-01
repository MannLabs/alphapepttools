import anndata as ad
import numpy as np
import pytest

from alphatools.pp import normalize
from alphatools.pp.norm import _total_mean_normalization, _validate_strategies


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
    """Generate count data (samples, features) where samples have different intensities"""
    # Reduce total intensity of sample 0
    return (
        np.array([[0.8, 1.0], [2.0, 0.0], [0.0, 2.0]]),
        np.array([[0.85925926, 1.07407407], [1.93333333, 0.0], [0.0, 1.93333333]]),
        np.array([1.07407407, 0.96666667, 0.96666667]),
    )


def test__validate_strategies() -> None:
    # Valid strategy
    _validate_strategies("total_mean")

    # Invalid strategy
    with pytest.raises(ValueError, match="`strategy` must be one of"):
        _validate_strategies("invalid_strategy")


def test__total_mean_normalization_all_equal(all_equal_count_data) -> None:
    array, norm_array_ref, norm_factors_ref = all_equal_count_data
    norm_array, norm_factors = _total_mean_normalization(array)

    assert np.isclose(norm_array, norm_array_ref, atol=1e-6).all()
    assert np.isclose(norm_factors, norm_factors_ref, atol=1e-6).all()


def test__mean_normalization_different(different_count_data) -> None:
    array, norm_array_ref, norm_factors_ref = different_count_data
    norm_array, norm_factors = _total_mean_normalization(array)

    assert np.isclose(norm_array, norm_array_ref, atol=1e-6).all()
    assert np.isclose(norm_factors, norm_factors_ref, atol=1e-6).all()


def test_normalize_default_parameters(different_count_data) -> None:
    """Test normalize with default parameters (normalizes adata.X in place)"""
    # Create test data
    array, norm_array_ref, _ = different_count_data
    adata = ad.AnnData(X=array.copy())

    # Normalize
    normalize(adata)

    # Check that X was normalized
    assert np.isclose(adata.X, norm_array_ref, atol=1e-6).all()
    assert len(adata.obs.columns) == 0
    assert len(adata.layers) == 0


def test_normalize_key_added(different_count_data) -> None:
    """Test normalize with key_added parameter"""
    # Create test data
    array, norm_array_ref, norm_factors_ref = different_count_data
    adata = ad.AnnData(X=array.copy())

    # Normalize
    normalize(adata, key_added="norm_factors")

    # Check that X was normalized
    assert np.isclose(adata.X, norm_array_ref, atol=1e-6).all()
    assert "norm_factors" in adata.obs.columns
    assert np.isclose(adata.obs["norm_factors"].to_numpy(), norm_factors_ref, atol=1e-6).all()
    assert len(adata.layers) == 0
