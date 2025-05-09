import numpy as np
import pytest
from anndata import AnnData

import alphatools as at


@pytest.fixture
def toy_adata():
    """Fixture to create a toy AnnData object for testing."""
    np.random.seed(42)
    data = np.random.randn(100, 20)  # 100 samples, 20 features
    var_names = [f"gene_{i}" for i in range(20)]
    obs_names = [f"cell_{i}" for i in range(100)]
    return AnnData(X=data, var={"var_names": var_names}, obs={"obs_names": obs_names})


# now test the PCA function
def test_run_pca(toy_adata):
    """Test the run_pca function on a toy dataset."""
    adata = at.pp.pca(toy_adata)

    # Assertions for Expected Outputs
    assert "X_pca" in adata.obsm, "PCA results not found in obsm"
    assert "pca" in adata.uns, "PCA metadata not found in uns"
    assert "PCs" in adata.varm, "Principal components not found in varm"

    # Check for API Changes
    required_attrs = {"X_pca", "pca", "PCs"}
    existing_attrs = set(adata.obsm.keys()).union(adata.uns.keys(), adata.varm.keys())
    missing_attrs = required_attrs - existing_attrs
    assert not missing_attrs, f"Scanpy API may have changed, missing attributes: {missing_attrs}"
