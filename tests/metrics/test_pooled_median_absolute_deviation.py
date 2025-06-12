from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.metrics import pooled_median_absolute_deviation
from alphatools.metrics._pmad import _pmad, _set_recursive_dict_keys


@pytest.fixture
def count_data_pmad() -> tuple[np.ndarray, float]:
    """Generate count data with known PMAD"""
    X = np.arange(0, 9, 1).reshape(3, 3)
    return X, 3.0


@pytest.fixture
def adata_pmad(count_data_pmad) -> tuple[np.ndarray, float]:
    """Generate count data with known PMAD"""
    # Concatenate the same count matrix with known PMADs for 3 different sample groups
    X, pmad = count_data_pmad
    n_obs = X.shape[0]

    sample_types = ["A"] * n_obs + ["B"] * n_obs + ["C"] * n_obs
    X = np.concatenate([X for _ in range(3)], axis=0)

    adata = ad.AnnData(X=X, obs=pd.DataFrame({"sample_type": sample_types}))

    return {"adata": adata, "pmad": {"A": pmad, "B": pmad, "C": pmad}, "group_key": "sample_type"}


@pytest.mark.parametrize(
    ("dictionary", "keys", "value", "reference"),
    [
        # Initial test
        ({}, ["key1"], "value", {"key1": "value"}),
        # Do not overwrite existing keys
        ({"existing_key": "existing_value"}, ["key1"], "value", {"existing_key": "existing_value", "key1": "value"}),
        # Multiple keys
        ({}, ["key1", "key2"], "value", {"key1": {"key2": "value"}}),
        # Write non-string values
        ({}, ["key1", "key2"], [], {"key1": {"key2": []}}),
    ],
)
def test_set_recursive_dict_keys(
    dictionary: dict[str, Any], value: Any, keys: list[str], reference: dict[str, Any]
) -> None:
    """Test recursively setting dictionary keys in a dictionary"""
    result = _set_recursive_dict_keys(dictionary=dictionary, keys=keys, value=value)

    assert result == reference


def test__pmad(count_data_pmad) -> None:
    X, pmad = count_data_pmad

    assert _pmad(x=X) == pmad


def test_pooled_median_absolute_deviation_return(adata_pmad) -> None:
    """Test if `pooled_median_absolute_deviation` computes group-wise PMAD correctly"""
    reference = pd.DataFrame.from_dict(adata_pmad["pmad"], orient="index", columns=["pmad"])

    pmad = pooled_median_absolute_deviation(adata_pmad["adata"], group_key=adata_pmad["group_key"], inplace=False)

    assert pmad.equals(reference)


def test_pooled_median_absolute_deviation_inplace(adata_pmad) -> None:
    """Test if `pooled_median_absolute_deviation` sets PMAD correctly in anndata object"""
    reference = adata_pmad["pmad"]
    adata = pooled_median_absolute_deviation(adata_pmad["adata"], group_key=adata_pmad["group_key"], inplace=True)

    assert adata.uns.get("metrics").get("pmad") == reference
