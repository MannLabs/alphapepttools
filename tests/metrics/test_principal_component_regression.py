import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.metrics import principal_component_regression


@pytest.fixture
def adata_dummy():
    """Minimal anndata object with PCA results."""
    np.random.seed(0)
    n_cells = 100
    n_genes = 50
    n_pcs = 10

    X = np.random.randn(n_cells, n_genes)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)])

    # Add covariates
    obs["continuous"] = np.random.randn(n_cells)
    obs["categorical"] = pd.Series(np.random.choice(["A", "B", "C"], size=n_cells), dtype="category")
    obs["string"] = ["label"] * n_cells  # unsupported type

    obsm = {"X_pca": np.random.randn(n_cells, n_pcs)}
    uns = {"pca": {"variance_ratio": np.linspace(0.1, 0.01, n_pcs)}}

    return ad.AnnData(
        X=X,
        obs=obs,
        obsm=obsm,
        uns=uns,
    )


def test_principal_component_regression_continuous(adata_dummy):
    score = principal_component_regression(adata_dummy, covariate="continuous")
    assert isinstance(score, float)
    assert score >= 0


def test_principal_component_regression_categorical(adata_dummy):
    score = principal_component_regression(adata_dummy, covariate="categorical")
    assert isinstance(score, float)
    assert score >= 0


def test_principal_component_regression_subset_components(adata_dummy):
    score_all = principal_component_regression(adata_dummy, covariate="continuous")
    score_subset = principal_component_regression(adata_dummy, covariate="continuous", n_components=5)
    assert score_subset <= score_all


def test_principal_component_regression_missing_covariate(adata_dummy):
    with pytest.raises(KeyError, match="not found in `adata.obs`"):
        principal_component_regression(adata_dummy, covariate="missing")


def test_principal_component_regression_missing_pca_key(adata_dummy):
    adata_dummy.obsm.pop("X_pca")
    with pytest.raises(KeyError, match="was not found in `adata.obsm`"):
        principal_component_regression(adata_dummy, covariate="continuous")


def test_principal_component_regression_missing_pca_uns_key(adata_dummy):
    adata_dummy.uns.pop("pca")
    with pytest.raises(KeyError, match="was not found in `adata.uns`"):
        principal_component_regression(adata_dummy, covariate="continuous")


def test_principal_component_regression_unsupported_dtype(adata_dummy):
    with pytest.raises(TypeError, match="not supported. Must be numeric or categorical"):
        principal_component_regression(adata_dummy, covariate="string")
