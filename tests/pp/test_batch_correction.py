import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy

from alphapepttools.pp.batch_correction import coerce_nans_to_batch, drop_singleton_batches, scanpy_pycombat
from alphapepttools.pp.impute import impute_gaussian


# Test AnnData wrapper for Scanpy's PyCombat implementation
@pytest.fixture
def pycombat_test_data_simple():
    def make_dummy_data():
        df = pd.DataFrame(
            {"A": [1, 2, 4, 8, 16, 128, 256, 512, 1024], "B": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
            index=list("ABCDEFGHI"),
            dtype=np.float64,
        )
        md = pd.DataFrame({"batch": list("xxxxxyyyy")}, index=list("ABCDEFGHI"))
        return df, md

    return make_dummy_data()


@pytest.fixture
def pycombat_test_data_complex():
    def make_dummy_data():
        df = pd.DataFrame(
            {"A": [1, 2, 4, 8, 16, 128, 256, 512, np.nan, 1024], "B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            index=list("ABCDEFGHIJ"),
            dtype=np.float64,
        )
        md = pd.DataFrame({"batch": [*list("xxx"), np.nan, np.nan, *list("yyyyz")]}, index=list("ABCDEFGHIJ"))
        return df, md

    return make_dummy_data()


def test_drop_singleton_batches():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [1, 1, 2, 2, 3]}, index=list("ABCDE"))
    md = pd.DataFrame({"batch": ["x", "x", "y", "y", "z"]}, index=list("ABCDE"))
    adata = ad.AnnData(df, obs=md)

    result = drop_singleton_batches(adata, "batch")

    expected_batch = ["x", "x", "y", "y"]
    assert result.obs["batch"].tolist() == expected_batch


def test_convert_na_to_batch():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]}, index=list("ABC"))
    md = pd.DataFrame({"batch": ["x", np.nan, "y"]}, index=list("ABC"))
    adata = ad.AnnData(df, obs=md)

    result = coerce_nans_to_batch(adata, "batch")

    expected_batch = ["x", "NA", "y"]
    assert result.obs["batch"].tolist() == expected_batch


@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("layer", [False, True])
@pytest.mark.parametrize(
    ("datatype"),
    [
        # Basic case: No NaNs, adequately sized batches
        ("simple"),
        # Complex case: NaNs in data and batch and singleton batches present
        ("complex"),
    ],
)
def test_scanpy_pycombat(datatype, pycombat_test_data_simple, pycombat_test_data_complex, layer: str, *, copy: bool):
    """
    Test the scanpy_pycombat function with various scenarios.
    """

    if datatype == "simple":
        df, md = pycombat_test_data_simple
    elif datatype == "complex":
        df, md = pycombat_test_data_complex

    adata = ad.AnnData(df, obs=md)
    adata = impute_gaussian(adata, copy=True)
    adata = drop_singleton_batches(adata, batch="batch")
    adata.layers["new_layer"] = adata.X.copy()

    # Compute expected results
    expected_adata = adata.copy()
    scanpy.pp.combat(expected_adata, key="batch", inplace=True)

    # Get comparison data from wrapper
    result = scanpy_pycombat(adata, batch="batch", layer=layer, copy=copy)

    if copy:
        assert isinstance(result, ad.AnnData)
        adata = result
    else:
        assert result is None

    pd.testing.assert_frame_equal(adata.to_df(layer=layer), expected_adata.to_df())
