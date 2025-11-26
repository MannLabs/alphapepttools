from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphapepttools.metrics import pooled_coefficient_of_variation, pooled_median_absolute_deviation
from alphapepttools.metrics._group_level import _pcv, _pmad, _set_nested_dict


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


@pytest.fixture
def count_data_pmad() -> tuple[np.ndarray, float]:
    """Generate count data with known PMAD"""
    X = np.arange(0, 9, 1).reshape(3, 3)
    return X, 3.0


class TestPmad:
    def test__pmad(self, count_data_pmad) -> None:
        X, pmad = count_data_pmad

        assert _pmad(x=X) == pmad


class TestPoooledMedianAbsoluteDeviation:
    @pytest.fixture
    def adata_pmad(self, count_data_pmad) -> tuple[np.ndarray, float]:
        """Generate count data with known PMAD"""
        # Concatenate the same count matrix with known PMADs for 3 different sample groups
        X, pmad = count_data_pmad
        n_obs = X.shape[0]

        sample_types = ["A"] * n_obs + ["B"] * n_obs + ["C"] * n_obs
        X = np.concatenate([X for _ in range(3)], axis=0)

        adata = ad.AnnData(X=X, layers={"layer": X}, obs=pd.DataFrame({"sample_type": sample_types}))

        return {"adata": adata, "pmad": {"A": pmad, "B": pmad, "C": pmad}, "group_key": "sample_type"}

    @pytest.mark.parametrize("layer", [None, "layer"])
    def test_pooled_median_absolute_deviation_return(self, adata_pmad: ad.AnnData, layer: str | None) -> None:
        """Test if `pooled_median_absolute_deviation` computes group-wise PMAD correctly"""
        reference = pd.DataFrame.from_dict(adata_pmad["pmad"], orient="index", columns=["pmad"])

        pmad = pooled_median_absolute_deviation(
            adata_pmad["adata"], group_key=adata_pmad["group_key"], layer=layer, inplace=False
        )

        pd.testing.assert_frame_equal(pmad, reference)

    @pytest.mark.parametrize("layer", [None, "layer"])
    def test_pooled_median_absolute_deviation_inplace(self, adata_pmad: ad.AnnData, layer: str | None) -> None:
        """Test if `pooled_median_absolute_deviation` sets PMAD correctly in anndata object"""
        reference = adata_pmad["pmad"]
        adata = adata_pmad["adata"].copy()

        pooled_median_absolute_deviation(adata, group_key=adata_pmad["group_key"], layer=layer, inplace=True)

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
        # Concatenate the same count matrix with known PMADs for 3 different sample groups
        X, pcv = count_data_pcv
        n_obs = X.shape[0]

        sample_types = ["A"] * n_obs + ["B"] * n_obs + ["C"] * n_obs
        X = np.concatenate([X for _ in range(3)], axis=0)

        adata = ad.AnnData(X=X, layers={"layer": X}, obs=pd.DataFrame({"sample_type": sample_types}))

        return {"adata": adata, "pcv": {"A": pcv, "B": pcv, "C": pcv}, "group_key": "sample_type"}

    @pytest.mark.parametrize("layer", [None, "layer"])
    def test_pooled_coefficient_of_variation_return(self, adata_pcv: ad.AnnData, layer: str | None) -> None:
        """Test if `pooled_median_absolute_deviation` computes group-wise PMAD correctly"""
        reference = pd.DataFrame.from_dict(adata_pcv["pcv"], orient="index", columns=["pcv"])

        pcv = pooled_coefficient_of_variation(
            adata_pcv["adata"], group_key=adata_pcv["group_key"], layer=layer, inplace=False
        )

        pd.testing.assert_frame_equal(pcv, reference, atol=1e-4)

    @pytest.mark.parametrize("layer", [None, "layer"])
    def test_pooled_coefficient_of_variation_inplace(self, adata_pcv: ad.AnnData, layer: str | None) -> None:
        """Test if `pooled_median_absolute_deviation` sets PMAD correctly in anndata object"""
        reference = adata_pcv["pcv"]
        adata = adata_pcv["adata"].copy()

        pooled_coefficient_of_variation(adata, group_key=adata_pcv["group_key"], layer=layer, inplace=True)
        result = adata.uns.get("metrics").get("pcv")

        assert all(key == ref_key for key, ref_key in zip(result.keys(), reference.keys(), strict=True))
        assert all(
            np.isclose(value, ref_value, atol=1e-4)
            for value, ref_value in zip(result.values(), reference.values(), strict=True)
        )
