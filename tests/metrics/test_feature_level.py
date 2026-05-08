"""Test feature-level metrics"""

import warnings
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphapepttools.metrics import (
    coefficient_of_variation,
    pooled_coefficient_of_variation,
    pooled_median_absolute_deviation,
)
from alphapepttools.metrics.feature_level import (
    _compute_pooled_groupwise_metric,
    _cv,
    _pcv,
    _pmad,
    _set_nested_dict,
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

    def test__cv_silent_on_empty_slices(self):
        """All-NaN columns must not raise nanmean/nanstd RuntimeWarnings."""
        # Column 0 has 2 valid values; column 1 is all-NaN.
        data = np.array([[1.0, np.nan], [2.0, np.nan]])

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning, message="Mean of empty slice")
            warnings.filterwarnings("error", category=RuntimeWarning, message="Degrees of freedom <= 0")
            result = _cv(data, min_valid=2, axis=0)

        assert np.isnan(result[1])  # all-NaN column → NaN result


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
def grouped_adata() -> tuple[ad.AnnData, pd.DataFrame]:
    """AnnData with two groups (A, B) and hand-computed per-feature, per-group CVs."""
    # Group A rows -> feature1: mean=2, std=sqrt(2/3); feature2: mean=2, std=sqrt(8/3)
    # Group B rows -> feature1: mean=4, std=sqrt(2/3); feature2: mean=2, std=sqrt(8/3)
    X = np.array(
        [
            [1.0, 0.0],
            [2.0, 2.0],
            [3.0, 4.0],
            [3.0, 0.0],
            [4.0, 2.0],
            [5.0, 4.0],
        ]
    )
    obs = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"]}, index=[f"s{i}" for i in range(6)])
    adata = ad.AnnData(X=X, layers={"layer": X * 2}, obs=obs)
    adata.var_names = ["feature1", "feature2"]

    expected = pd.DataFrame(
        {
            "A": [np.std(X[:3, 0]) / np.mean(X[:3, 0]), np.std(X[:3, 1]) / np.mean(X[:3, 1])],
            "B": [np.std(X[3:, 0]) / np.mean(X[3:, 0]), np.std(X[3:, 1]) / np.mean(X[3:, 1])],
        },
        index=adata.var_names,
    )
    return adata, expected


class TestCoefficientOfVariationGrouped:
    @pytest.mark.parametrize("layer", [None, "layer"])
    def test_grouped_writes_to_varm(self, grouped_adata, layer: str | None) -> None:
        """Grouped CV writes a DataFrame to adata.varm[key_added] with shape (n_features, n_groups)"""
        adata, expected = grouped_adata
        # Layer is X*2, so the CV (scale-invariant) is identical to the X-based expectation.
        coefficient_of_variation(adata, group_column="group", layer=layer)

        assert "cv" in adata.varm
        result = adata.varm["cv"]
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["A", "B"]
        assert list(result.index) == list(adata.var_names)
        np.testing.assert_allclose(result.values, expected.values, atol=1e-6)
        # Ungrouped slot is untouched
        assert "cv" not in adata.var.columns

    def test_grouped_custom_key(self, grouped_adata) -> None:
        """key_added controls the varm key when grouped"""
        adata, _ = grouped_adata
        coefficient_of_variation(adata, group_column="group", key_added="my_cv")

        assert "my_cv" in adata.varm
        assert "cv" not in adata.varm

    def test_grouped_copy_does_not_modify_original(self, grouped_adata) -> None:
        """copy=True returns a modified copy and leaves the original alone"""
        adata, _ = grouped_adata
        result = coefficient_of_variation(adata, group_column="group", copy=True)

        assert result is not None
        assert "cv" in result.varm
        assert "cv" not in adata.varm

    def test_grouped_raises_on_nan_group(self, grouped_adata) -> None:
        """NaN values in group_column raise ValueError"""
        adata, _ = grouped_adata
        adata.obs["group"] = adata.obs["group"].astype(object)
        adata.obs.loc[adata.obs.index[0], "group"] = np.nan

        with pytest.raises(ValueError, match="contains NaNs"):
            coefficient_of_variation(adata, group_column="group")

    def test_grouped_min_valid_propagates(self, grouped_adata) -> None:
        """Groups smaller than min_valid produce NaN CVs for all features in that group"""
        adata, _ = grouped_adata
        # Each group has 3 samples; setting min_valid=4 forces NaN everywhere
        coefficient_of_variation(adata, group_column="group", min_valid=4)

        result = adata.varm["cv"]
        assert result.isna().all().all()


class TestSetNestedDict:
    @pytest.mark.parametrize(
        ("dictionary", "keys", "value", "reference"),
        [
            # Initial test
            ({}, ["key1"], "value", {"key1": "value"}),
            # Do not overwrite existing keys
            (
                {"existing_key": "existing_value"},
                ["key1"],
                "value",
                {"existing_key": "existing_value", "key1": "value"},
            ),
            # Multiple keys
            ({}, ["key1", "key2"], "value", {"key1": {"key2": "value"}}),
            # Write non-string values
            ({}, ["key1", "key2"], [], {"key1": {"key2": []}}),
        ],
    )
    def test__set_nested_dict(
        self, dictionary: dict[str, Any], value: Any, keys: list[str], reference: dict[str, Any]
    ) -> None:
        """Test recursively setting dictionary keys in a dictionary"""
        result = _set_nested_dict(dictionary=dictionary, keys=keys, value=value)

        assert result == reference


class TestComputePooledGroupwiseMetric:
    """Test that aggregation function returns correct values"""

    @pytest.fixture
    def adata_grouped(self) -> ad.AnnData:
        """AnnData with two groups, each with distinct values"""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]])
        obs = pd.DataFrame({"group": ["A", "A", "B", "B"]})
        return ad.AnnData(X=X, layers={"layer": X * 2}, obs=obs)

    def test__compute_pooled_groupwise_metric(self, adata_grouped: ad.AnnData) -> None:
        """Test groupwise metric computation with a simple mean function"""
        result = _compute_pooled_groupwise_metric(adata_grouped, func=lambda x: np.mean(x), group_column="group")
        # Group A: mean([[1,2],[3,4]]) = 2.5, Group B: mean([[10,20],[30,40]]) = 25.0
        assert result == {"A": 2.5, "B": 25.0}

    def test__compute_pooled_groupwise_metric_layer(self, adata_grouped: ad.AnnData) -> None:
        """Test groupwise metric computation using a layer"""
        result = _compute_pooled_groupwise_metric(
            adata_grouped, func=lambda x: np.mean(x), group_column="group", layer="layer"
        )
        assert result == {"A": 5.0, "B": 50.0}

    def test__compute_pooled_groupwise_metric_kwargs(self, adata_grouped: ad.AnnData) -> None:
        """Test that kwargs are passed to the aggregation function"""
        result = _compute_pooled_groupwise_metric(
            adata_grouped, func=lambda _, return_value: return_value, group_column="group", return_value=10.0
        )
        assert result == {"A": 10.0, "B": 10.0}

    def test__compute_pooled_groupwise_metric_invalid_return_type(self, adata_grouped: ad.AnnData) -> None:
        """Test that non-float return raises TypeError"""
        with pytest.raises(TypeError, match="needs to return a numeric value"):
            _compute_pooled_groupwise_metric(adata_grouped, func=lambda _: "not a number", group_column="group")


@pytest.fixture
def count_data_pmad() -> tuple[np.ndarray, float]:
    """Generate count data with known PMAD"""
    X = np.arange(0, 9, 1).reshape(3, 3)
    return X, 3.0


class TestPmad:
    def test__pmad(self, count_data_pmad) -> None:
        X, pmad = count_data_pmad

        assert _pmad(x=X) == pmad


class TestPooledMedianAbsoluteDeviation:
    @pytest.fixture
    def adata_pmad(self, count_data_pmad) -> tuple[np.ndarray, float]:
        """Generate count data with known PMAD"""
        # Concatenate the same count matrix with known PMADs for 3 different sample groups
        X, pmad = count_data_pmad
        n_obs = X.shape[0]

        sample_types = ["A"] * n_obs + ["B"] * n_obs + ["C"] * n_obs
        X = np.concatenate([X for _ in range(3)], axis=0)

        adata = ad.AnnData(X=X, layers={"layer": X}, obs=pd.DataFrame({"sample_type": sample_types}))

        return {"adata": adata, "pmad": {"A": pmad, "B": pmad, "C": pmad}, "group_column": "sample_type"}

    @pytest.mark.parametrize("layer", [None, "layer"])
    def test_pooled_median_absolute_deviation_return(self, adata_pmad: ad.AnnData, layer: str | None) -> None:
        """Test if `pooled_median_absolute_deviation` computes group-wise PMAD correctly"""
        reference = pd.DataFrame.from_dict(adata_pmad["pmad"], orient="index", columns=["pmad"])

        pmad = pooled_median_absolute_deviation(
            adata_pmad["adata"], group_column=adata_pmad["group_column"], layer=layer, inplace=False
        )

        pd.testing.assert_frame_equal(pmad, reference)

    @pytest.mark.parametrize("layer", [None, "layer"])
    def test_pooled_median_absolute_deviation_inplace(self, adata_pmad: ad.AnnData, layer: str | None) -> None:
        """Test if `pooled_median_absolute_deviation` sets PMAD correctly in anndata object"""
        reference = adata_pmad["pmad"]
        adata = adata_pmad["adata"].copy()

        pooled_median_absolute_deviation(adata, group_column=adata_pmad["group_column"], layer=layer, inplace=True)

        assert adata.uns.get("metrics").get("pmad") == reference


@pytest.fixture
def count_data_pcv() -> tuple[np.ndarray, float]:
    """Generate count data with known PCV"""
    # STD: 0.5, MEAN: 0.5, 1.5, 2.5, 3.5
    # CVs: 1, 1/3, 1/5, 1/7
    # mean feature-wise CV: (1 + 1/3 + 1/5 + 1/7)/4
    X = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 3], [1, 2, 3, 4]])
    return X, 0.419047


class TestPcv:
    def test__pcv(self, count_data_pcv) -> None:
        X, pcv = count_data_pcv

        result = _pcv(x=X, min_valid=3)
        assert np.allclose(result, pcv, atol=1e-4)


class TestPooledCoefficientOfVariation:
    @pytest.fixture
    def adata_pcv(self, count_data_pcv) -> tuple[np.ndarray, float]:
        """Generate count data with known PCV"""
        # Concatenate the same count matrix with known PCVs for 3 different sample groups
        X, pcv = count_data_pcv
        n_obs = X.shape[0]

        sample_types = ["A"] * n_obs + ["B"] * n_obs + ["C"] * n_obs
        X = np.concatenate([X for _ in range(3)], axis=0)

        adata = ad.AnnData(X=X, layers={"layer": X}, obs=pd.DataFrame({"sample_type": sample_types}))

        return {"adata": adata, "pcv": {"A": pcv, "B": pcv, "C": pcv}, "group_column": "sample_type"}

    @pytest.mark.parametrize("layer", [None, "layer"])
    def test_pooled_coefficient_of_variation_return(self, adata_pcv: ad.AnnData, layer: str | None) -> None:
        """Test if `pooled_coefficient_of_variation` computes group-wise PCV correctly"""
        reference = pd.DataFrame.from_dict(adata_pcv["pcv"], orient="index", columns=["pcv"])

        pcv = pooled_coefficient_of_variation(
            adata_pcv["adata"], group_column=adata_pcv["group_column"], layer=layer, inplace=False
        )

        pd.testing.assert_frame_equal(pcv, reference, atol=1e-4)

    @pytest.mark.parametrize("layer", [None, "layer"])
    def test_pooled_coefficient_of_variation_inplace(self, adata_pcv: ad.AnnData, layer: str | None) -> None:
        """Test if `pooled_coefficient_of_variation` sets PCV correctly in anndata object"""
        reference = adata_pcv["pcv"]
        adata = adata_pcv["adata"].copy()

        pooled_coefficient_of_variation(adata, group_column=adata_pcv["group_column"], layer=layer, inplace=True)
        result = adata.uns.get("metrics").get("pcv")

        assert all(key == ref_key for key, ref_key in zip(result.keys(), reference.keys(), strict=True))
        assert all(
            np.isclose(value, ref_value, atol=1e-4)
            for value, ref_value in zip(result.values(), reference.values(), strict=True)
        )
