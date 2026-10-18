import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy
from anndata.tests.helpers import assert_adata_equal

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


@pytest.mark.parametrize("batch_dtype", [object, "category"])
def test_convert_na_to_batch(batch_dtype):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]}, index=list("ABC"))
    md = pd.DataFrame({"batch": ["x", np.nan, "y"]}, index=list("ABC")).astype(batch_dtype)
    adata = ad.AnnData(df, obs=md)

    result = coerce_nans_to_batch(adata, "batch")

    expected_batch = ["x", "NA", "y"]
    assert result.obs["batch"].tolist() == expected_batch


def _expected_combat(adata: ad.AnnData, batch: str, covariates: list[str] | None, layer: str | None) -> ad.AnnData:
    """Reference result: copy of `adata` where only the target matrix is replaced by its scanpy.pp.combat correction."""
    proxy = coerce_nans_to_batch(ad.AnnData(X=adata.to_df(layer=layer).to_numpy(), obs=adata.obs.copy()), batch)
    corrected = scanpy.pp.combat(proxy, key=batch, covariates=covariates, inplace=False)

    expected = adata.copy()
    if layer is None:
        expected.X = corrected
    else:
        expected.layers[layer] = corrected
    return expected


@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("layer", [None, "new_layer"])
@pytest.mark.parametrize("covariates", [None])
@pytest.mark.parametrize(
    ("datatype"),
    [
        # Basic case: No NaNs, adequately sized batches
        ("simple"),
        # Complex case: NaNs in data and batch and singleton batches present
        ("complex"),
    ],
)
def test_scanpy_pycombat(
    datatype, pycombat_test_data_simple, pycombat_test_data_complex, covariates, layer: str | None, *, copy: bool
):
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
    # Layer must differ from X, otherwise correcting the wrong matrix goes unnoticed
    adata.layers["new_layer"] = adata.X * 2 + 3
    original = adata.copy()

    expected = _expected_combat(original, batch="batch", covariates=covariates, layer=layer)

    result = scanpy_pycombat(adata, batch="batch", covariates=covariates, layer=layer, copy=copy)

    if copy:
        assert isinstance(result, ad.AnnData)
        assert_adata_equal(adata, original)
        adata = result
    else:
        assert result is None

    assert_adata_equal(adata, expected)


@pytest.mark.parametrize("batch_dtype", [object, "category"])
@pytest.mark.parametrize("copy", [False, True])
def test_scanpy_pycombat_nan_batch(pycombat_test_data_simple, batch_dtype, *, copy: bool):
    """NaNs in the batch column are corrected as a single 'NA' batch, also when modifying inplace."""
    df, _ = pycombat_test_data_simple
    md = pd.DataFrame({"batch": ["x", "x", "x", np.nan, np.nan, "y", "y", "y", "y"]}, index=df.index).astype(
        batch_dtype
    )
    adata = ad.AnnData(df, obs=md)
    original = adata.copy()

    expected = _expected_combat(original, batch="batch", covariates=None, layer=None)

    result = scanpy_pycombat(adata, batch="batch", copy=copy)
    adata = result if copy else adata

    assert_adata_equal(adata, expected)
