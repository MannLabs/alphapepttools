"""Test feature-level metrics"""

import anndata as ad
import numpy as np
import pytest

from alphapepttools.metrics import coefficient_of_variation
from alphapepttools.metrics.feature_level import (
    _cv,
    calculate_qc_metrics,
    fraction_complete,
    number_detected,
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
        assert np.allclose(qc_adata.obs["total_intensity"].values, expected)

    def test_total_intensity_custom_col_name(self, qc_adata):
        """Test total_intensity with custom column name."""
        total_intensity(qc_adata, column="custom_total_intensity")

        assert "custom_total_intensity" in qc_adata.obs.columns
        assert "total_intensity" not in qc_adata.obs.columns

    def test_total_intensity_return_value(self, qc_adata):
        """Test total_intensity returns values when inplace=False."""
        expected = np.array([6.0, 2.0, 1.0])

        result = total_intensity(qc_adata, inplace=False)

        assert result is not None
        assert np.allclose(result, expected)
        assert "total_intensity" not in qc_adata.obs.columns

    @pytest.mark.parametrize("layer", [None, "raw"])
    def test_total_intensity_layer(self, qc_adata, layer):
        """Test total_intensity works with different layers."""
        total_intensity(qc_adata, layer=layer)

        assert "total_intensity" in qc_adata.obs.columns

    def test_total_intensity_invalid_layer(self, qc_adata):
        """Test total_intensity raises error for invalid layer."""
        with pytest.raises(ValueError, match="not found in adata.layers"):
            total_intensity(qc_adata, layer="nonexistent")

    def test_total_intensity_axis_var_expected_values(self, qc_adata):
        """Test total_intensity with axis='var' computes correct values."""
        expected = np.array([2.0, 4.0, 3.0])

        total_intensity(qc_adata, axis="var")

        assert "total_intensity" in qc_adata.var.columns
        assert np.allclose(qc_adata.var["total_intensity"].values, expected)

    def test_total_intensity_axis_var_custom_col_name(self, qc_adata):
        """Test total_intensity with axis='var' and custom column name."""
        total_intensity(qc_adata, axis="var", column="custom_total")

        assert "custom_total" in qc_adata.var.columns
        assert "total_intensity" not in qc_adata.var.columns

    def test_total_intensity_axis_var_return_value(self, qc_adata):
        """Test total_intensity with axis='var' returns values when inplace=False."""
        expected = np.array([2.0, 4.0, 3.0])

        result = total_intensity(qc_adata, axis="var", inplace=False)

        assert result is not None
        assert np.allclose(result, expected)
        assert "total_intensity" not in qc_adata.var.columns

    def test_total_intensity_invalid_axis(self, qc_adata):
        """Test total_intensity raises error for invalid axis."""
        with pytest.raises(ValueError, match="axis must be 'obs' or 'var'"):
            total_intensity(qc_adata, axis="invalid")


class TestNumberDetected:
    def test_number_detected_expected_values(self, qc_adata):
        """Test number_detected computes correct values."""
        # detected = not (NaN, zero, negative, inf)
        # Row 0: [1, 2, 3] -> 3 detected
        # Row 1: [0, 2, nan] -> 1 detected
        # Row 2: [1, 0, 0] -> 1 detected
        expected = np.array([3, 1, 1])

        number_detected(qc_adata)

        assert "number_detected" in qc_adata.obs.columns
        assert np.array_equal(qc_adata.obs["number_detected"].values, expected)

    def test_number_detected_custom_col_name(self, qc_adata):
        """Test number_detected with custom column name."""
        number_detected(qc_adata, column="custom_num")

        assert "custom_num" in qc_adata.obs.columns
        assert "number_detected" not in qc_adata.obs.columns

    def test_number_detected_return_value(self, qc_adata):
        """Test number_detected returns values when inplace=False."""
        expected = np.array([3, 1, 1])

        result = number_detected(qc_adata, inplace=False)

        assert result is not None
        assert np.array_equal(result, expected)
        assert "number_detected" not in qc_adata.obs.columns

    @pytest.mark.parametrize("layer", [None, "raw"])
    def test_number_detected_layer(self, qc_adata, layer):
        """Test number_detected works with different layers."""
        number_detected(qc_adata, layer=layer)

        assert "number_detected" in qc_adata.obs.columns

    def test_number_detected_invalid_layer(self, qc_adata):
        """Test number_detected raises error for invalid layer."""
        with pytest.raises(ValueError, match="not found in adata.layers"):
            number_detected(qc_adata, layer="nonexistent")

    def test_number_detected_axis_var_expected_values(self, qc_adata):
        """Test number_detected with axis='var' computes correct values."""
        # Col 0: [1, 0, 1] -> 2 detected
        # Col 1: [2, 2, 0] -> 2 detected
        # Col 2: [3, nan, 0] -> 1 detected
        expected = np.array([2, 2, 1])

        number_detected(qc_adata, axis="var")

        assert "number_detected" in qc_adata.var.columns
        assert np.array_equal(qc_adata.var["number_detected"].values, expected)

    def test_number_detected_axis_var_custom_col_name(self, qc_adata):
        """Test number_detected with axis='var' and custom column name."""
        number_detected(qc_adata, axis="var", column="custom_num")

        assert "custom_num" in qc_adata.var.columns
        assert "number_detected" not in qc_adata.var.columns

    def test_number_detected_axis_var_return_value(self, qc_adata):
        """Test number_detected with axis='var' returns values when inplace=False."""
        expected = np.array([2, 2, 1])

        result = number_detected(qc_adata, axis="var", inplace=False)

        assert result is not None
        assert np.array_equal(result, expected)
        assert "number_detected" not in qc_adata.var.columns

    def test_number_detected_invalid_axis(self, qc_adata):
        """Test number_detected raises error for invalid axis."""
        with pytest.raises(ValueError, match="axis must be 'obs' or 'var'"):
            number_detected(qc_adata, axis="invalid")


class TestFractionComplete:
    def test_fraction_complete_expected_values(self, qc_adata):
        """Test fraction_complete computes correct values."""
        # Row 0: 3/3 = 1.0
        # Row 1: 1/3 = 0.333...
        # Row 2: 1/3 = 0.333...
        expected = np.array([1.0, 1 / 3, 1 / 3])

        fraction_complete(qc_adata)

        assert "fraction_complete" in qc_adata.obs.columns
        assert np.allclose(qc_adata.obs["fraction_complete"].values, expected)

    def test_fraction_complete_custom_col_name(self, qc_adata):
        """Test fraction_complete with custom column name."""
        fraction_complete(qc_adata, column="custom_frac")

        assert "custom_frac" in qc_adata.obs.columns
        assert "fraction_complete" not in qc_adata.obs.columns

    def test_fraction_complete_return_value(self, qc_adata):
        """Test fraction_complete returns values when inplace=False."""
        expected = np.array([1.0, 1 / 3, 1 / 3])

        result = fraction_complete(qc_adata, inplace=False)

        assert result is not None
        assert np.allclose(result, expected)
        assert "fraction_complete" not in qc_adata.obs.columns

    @pytest.mark.parametrize("layer", [None, "raw"])
    def test_fraction_complete_layer(self, qc_adata, layer):
        """Test fraction_complete works with different layers."""
        fraction_complete(qc_adata, layer=layer)

        assert "fraction_complete" in qc_adata.obs.columns

    def test_fraction_complete_axis_var_expected_values(self, qc_adata):
        """Test fraction_complete with axis='var' computes correct values."""
        # Col 0: [1, 0, 1] -> 2 detected out of 3 obs -> 2/3
        # Col 1: [2, 2, 0] -> 2 detected out of 3 obs -> 2/3
        # Col 2: [3, nan, 0] -> 1 detected out of 3 obs -> 1/3
        expected = np.array([2 / 3, 2 / 3, 1 / 3])

        fraction_complete(qc_adata, axis="var")

        assert "fraction_complete" in qc_adata.var.columns
        assert np.allclose(qc_adata.var["fraction_complete"].values, expected)

    def test_fraction_complete_axis_var_custom_col_name(self, qc_adata):
        """Test fraction_complete with axis='var' and custom column name."""
        fraction_complete(qc_adata, axis="var", column="custom_frac")

        assert "custom_frac" in qc_adata.var.columns
        assert "fraction_complete" not in qc_adata.var.columns

    def test_fraction_complete_axis_var_return_value(self, qc_adata):
        """Test fraction_complete with axis='var' returns values when inplace=False."""
        expected = np.array([2 / 3, 2 / 3, 1 / 3])

        result = fraction_complete(qc_adata, axis="var", inplace=False)

        assert result is not None
        assert np.allclose(result, expected)
        assert "fraction_complete" not in qc_adata.var.columns

    def test_fraction_complete_invalid_axis(self, qc_adata):
        """Test fraction_complete raises error for invalid axis."""
        with pytest.raises(ValueError, match="axis must be 'obs' or 'var'"):
            fraction_complete(qc_adata, axis="invalid")

    def test_fraction_complete_invalid_layer(self, qc_adata):
        """Test fraction_complete raises error for invalid layer."""
        with pytest.raises(ValueError, match="not found in adata.layers"):
            fraction_complete(qc_adata, layer="nonexistent")


class TestCalculateQCMetrics:
    def test_calculate_qc_metrics_adds_all_columns(self, qc_adata):
        """Test calculate_qc_metrics adds all expected columns."""
        calculate_qc_metrics(qc_adata)

        # obs columns
        assert "total_sample_intensity" in qc_adata.obs.columns
        assert "num_features_detected" in qc_adata.obs.columns
        assert "fraction_detected_features" in qc_adata.obs.columns
        # var columns
        assert "total_feature_intensity" in qc_adata.var.columns
        assert "num_samples_detected" in qc_adata.var.columns
        assert "fraction_detected_samples" in qc_adata.var.columns

    def test_calculate_qc_metrics_correct_values(self, qc_adata):
        """Test calculate_qc_metrics computes correct values."""
        # obs expected values
        expected_total_obs = np.array([6.0, 2.0, 1.0])
        expected_num_obs = np.array([3, 1, 1])
        expected_frac_obs = np.array([1.0, 1 / 3, 1 / 3])
        # var expected values
        expected_total_var = np.array([2.0, 4.0, 3.0])
        expected_num_var = np.array([2, 2, 1])
        expected_frac_var = np.array([2 / 3, 2 / 3, 1 / 3])

        calculate_qc_metrics(qc_adata)

        # obs assertions
        assert np.allclose(qc_adata.obs["total_sample_intensity"].values, expected_total_obs)
        assert np.array_equal(qc_adata.obs["num_features_detected"].values, expected_num_obs)
        assert np.allclose(qc_adata.obs["fraction_detected_features"].values, expected_frac_obs)
        # var assertions
        assert np.allclose(qc_adata.var["total_feature_intensity"].values, expected_total_var)
        assert np.array_equal(qc_adata.var["num_samples_detected"].values, expected_num_var)
        assert np.allclose(qc_adata.var["fraction_detected_samples"].values, expected_frac_var)

    @pytest.mark.parametrize("layer", [None, "raw"])
    def test_calculate_qc_metrics_layer(self, qc_adata, layer):
        """Test calculate_qc_metrics works with different layers."""
        calculate_qc_metrics(qc_adata, layer=layer)

        # obs columns
        assert "total_sample_intensity" in qc_adata.obs.columns
        assert "num_features_detected" in qc_adata.obs.columns
        assert "fraction_detected_features" in qc_adata.obs.columns
        # var columns
        assert "total_feature_intensity" in qc_adata.var.columns
        assert "num_samples_detected" in qc_adata.var.columns
        assert "fraction_detected_samples" in qc_adata.var.columns
