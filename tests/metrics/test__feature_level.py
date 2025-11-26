"""Test feature-level metrics"""

import anndata as ad
import numpy as np
import pytest

from alphapepttools.metrics import coefficient_of_variation
from alphapepttools.metrics._feature_level import _cv


@pytest.fixture
def example_data():
    """Example data with ground truth"""
    data = np.array(
        [[1, 1, np.nan, 1, 0], [1, 2, np.nan, 2, 0], [1, 1, np.nan, 1, 0], [1, 2, 1, 2, 0], [1, np.nan, 1, 1.5, 0]]
    )

    cv = {
        3: np.array([0, 0.3333, np.nan, 0.2981, np.nan]),
        4: np.array([0, 0.3333, np.nan, 0.2981, np.nan]),
        5: np.array([0, np.nan, np.nan, 0.2981, np.nan]),
    }

    return data, cv


@pytest.fixture
def example_adata(example_data) -> ad.AnnData:
    data, cv = example_data
    return ad.AnnData(X=data, layers={"layer": data}), cv


class TestCV:
    @pytest.mark.parametrize("min_valid", [3, 4, 5])
    def test__cv(self, example_data, min_valid: int):
        """Test computation of coefficient of variation"""
        data, cv_dict = example_data
        ground_truth = cv_dict[min_valid]

        results = _cv(data=data, min_valid=min_valid, axis=0)

        assert np.allclose(ground_truth, results, atol=0.001, equal_nan=True)


class TestCoefficientOfVariation:
    @pytest.mark.parametrize("min_valid", [3, 4, 5])
    @pytest.mark.parametrize("copy", [True, False])
    @pytest.mark.parametrize("layer", [None, "layer"])
    def test_coefficient_of_variation(self, example_adata, layer: str | None, min_valid: int, *, copy: bool):
        """Test CV computation for data with nan values"""
        adata, cv = example_adata
        expected_cv = cv[min_valid]

        result = coefficient_of_variation(adata, layer=layer, min_valid=min_valid, copy=copy)

        if copy:
            assert isinstance(result, ad.AnnData)
            adata_result = result
        else:
            assert result is None
            adata_result = adata

        assert "cv" in adata_result.var.columns
        assert np.allclose(adata_result.var["cv"].values, expected_cv, atol=0.001, equal_nan=True)

    @pytest.mark.parametrize("min_valid", [3])
    def test_coefficient_of_variation_custom_key(self, example_adata, min_valid: int):
        """Test CV computation with custom key name"""
        adata, cv = example_adata
        expected_cv = cv[min_valid]

        coefficient_of_variation(adata, key_added="custom_cv", min_valid=min_valid)

        assert "custom_cv" in adata.var.columns
        assert "cv" not in adata.var.columns
        assert np.allclose(adata.var["custom_cv"].values, expected_cv, atol=0.001, equal_nan=True)
