"""Test feature-level metrics"""

import anndata as ad
import numpy as np
import pytest

from alphapepttools.metrics import coefficient_of_variation
from alphapepttools.metrics._feature_level import (
    _cv,
    calculate_qc_metrics,
    fraction_complete,
    num_detected,
    total_intensity,
)


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


@pytest.fixture
def qc_adata():
    """Example data for QC metrics testing."""
    # Data with NaN, zeros, and valid values
    # Row 0: [1, 2, 3] -> sum=6, detected=3, frac=1.0
    # Row 1: [0, 2, np.nan] -> sum=2, detected=1, frac=1/3
    # Row 2: [1, 0, 0] -> sum=1, detected=1, frac=1/3
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 2.0, np.nan],
            [1.0, 0.0, 0.0],
        ]
    )
    return ad.AnnData(X=data, layers={"raw": data.copy()})


class TestTotalIntensity:
    def test_total_intensity_expected_values(self, qc_adata):
        """Test total_intensity computes correct values."""
        expected = np.array([6.0, 2.0, 1.0])

        total_intensity(qc_adata)

        assert "total_intensity" in qc_adata.obs.columns
        assert np.allclose(qc_adata.obs["total_intensity"].values, expected, equal_nan=True)

    def test_total_intensity_custom_col_name(self, qc_adata):
        """Test total_intensity with custom column name."""
        total_intensity(qc_adata, obs_col_name="custom_total_intensity")
        assert "custom_total_intensity" in qc_adata.obs.columns
        assert "total_intensity" not in qc_adata.obs.columns

    def test_total_intensity_return_value(self, qc_adata):
        """Test total_intensity returns values when add_to_adata=False."""
        expected = np.array([6.0, 2.0, 1.0])

        result = total_intensity(qc_adata, inplace=False)

        assert result is not None
        assert np.allclose(result, expected, equal_nan=True)
        assert "total_intensity" not in qc_adata.obs.columns

    @pytest.mark.parametrize("layer", [None, "raw"])
    def test_total_intensity_layer(self, qc_adata, layer):
        """Test total_intensity works with different layers."""
        total_intensity(qc_adata, layer=layer)

        assert "total_intensity" in qc_adata.obs.columns


class TestNumDetected:
    def test_num_detected_expected_values(self, qc_adata):
        """Test num_detected computes correct values."""
        # detected = not (NaN, zero, negative, inf)
        # Row 0: [1, 2, 3] -> 3 detected
        # Row 1: [0, 2, nan] -> 1 detected
        # Row 2: [1, 0, 0] -> 1 detected
        expected = np.array([3, 1, 1])

        num_detected(qc_adata)

        assert "num_prot" in qc_adata.obs.columns
        assert np.array_equal(qc_adata.obs["num_prot"].values, expected)

    def test_num_detected_custom_col_name(self, qc_adata):
        """Test num_detected with custom column name."""
        num_detected(qc_adata, obs_col_name="custom_num")

        assert "custom_num" in qc_adata.obs.columns
        assert "num_prot" not in qc_adata.obs.columns

    def test_num_detected_return_value(self, qc_adata):
        """Test num_detected returns values when add_to_adata=False."""
        expected = np.array([3, 1, 1])

        result = num_detected(qc_adata, inplace=False)

        assert result is not None
        assert np.array_equal(result, expected)
        assert "num_prot" not in qc_adata.obs.columns

    @pytest.mark.parametrize("layer", [None, "raw"])
    def test_num_detected_layer(self, qc_adata, layer):
        """Test num_detected works with different layers."""
        num_detected(qc_adata, layer=layer)

        assert "num_prot" in qc_adata.obs.columns

    def test_num_detected_invalid_layer(self, qc_adata):
        """Test num_detected raises error for invalid layer."""
        with pytest.raises(ValueError, match="not found in adata.layers"):
            num_detected(qc_adata, layer="nonexistent")


class TestFracDetected:
    def test_frac_detected_expected_values(self, qc_adata):
        """Test frac_detected computes correct values."""
        # Row 0: 3/3 = 1.0
        # Row 1: 1/3 = 0.333...
        # Row 2: 1/3 = 0.333...
        expected = np.array([1.0, 1 / 3, 1 / 3])

        fraction_complete(qc_adata)

        assert "fraction_complete" in qc_adata.obs.columns
        assert np.allclose(qc_adata.obs["fraction_complete"].values, expected)

    def test_frac_detected_custom_col_name(self, qc_adata):
        """Test frac_detected with custom column name."""
        fraction_complete(qc_adata, obs_col_name="custom_frac")

        assert "custom_frac" in qc_adata.obs.columns
        assert "fraction_complete" not in qc_adata.obs.columns

    def test_frac_detected_return_value(self, qc_adata):
        """Test frac_detected returns values when add_to_adata=False."""
        expected = np.array([1.0, 1 / 3, 1 / 3])

        result = fraction_complete(qc_adata, inplace=False)

        assert result is not None
        assert np.allclose(result, expected)
        assert "fraction_complete" not in qc_adata.obs.columns

    @pytest.mark.parametrize("layer", [None, "raw"])
    def test_frac_detected_layer(self, qc_adata, layer):
        """Test frac_detected works with different layers."""
        fraction_complete(qc_adata, layer=layer)

        assert "fraction_complete" in qc_adata.obs.columns


class TestCalculateQCMetrics:
    def test_calculate_qc_metrics_adds_all_columns(self, qc_adata):
        """Test calculate_qc_metrics adds all expected columns."""
        calculate_qc_metrics(qc_adata)

        assert "total_intensity" in qc_adata.obs.columns
        assert "num_prot" in qc_adata.obs.columns
        assert "fraction_complete" in qc_adata.obs.columns

    def test_calculate_qc_metrics_correct_values(self, qc_adata):
        """Test calculate_qc_metrics computes correct values."""
        expected_sum = np.array([6.0, 2.0, 1.0])
        expected_num = np.array([3, 1, 1])
        expected_frac = np.array([1.0, 1 / 3, 1 / 3])

        calculate_qc_metrics(qc_adata)

        assert np.allclose(qc_adata.obs["total_intensity"].values, expected_sum, equal_nan=True)
        assert np.array_equal(qc_adata.obs["num_prot"].values, expected_num)
        assert np.allclose(qc_adata.obs["fraction_complete"].values, expected_frac)

    @pytest.mark.parametrize("layer", [None, "raw"])
    def test_calculate_qc_metrics_layer(self, qc_adata, layer):
        """Test calculate_qc_metrics works with different layers."""
        calculate_qc_metrics(qc_adata, layer=layer)

        assert "total_intensity" in qc_adata.obs.columns
        assert "num_prot" in qc_adata.obs.columns
        assert "fraction_complete" in qc_adata.obs.columns
