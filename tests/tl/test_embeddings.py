import numpy as np
import pytest
from anndata import AnnData

import alphapepttools as at


@pytest.fixture
def toy_adata():
    """Fixture to create a toy AnnData object for testing."""
    np.random.seed(42)
    data = np.random.randn(100, 20)  # 100 samples, 20 features
    var_names = [f"gene_{i}" for i in range(20)]
    obs_names = [f"cell_{i}" for i in range(100)]
    return AnnData(X=data, var={"var_names": var_names}, obs={"obs_names": obs_names})


@pytest.fixture
def toy_adata_with_layers(toy_adata):
    """Fixture to create a toy AnnData object with layers for testing."""
    toy_adata.layers["norm"] = toy_adata.X.copy() * 1.5
    toy_adata.layers["scaled"] = toy_adata.X.copy() * 0.5
    return toy_adata


@pytest.fixture
def toy_adata_with_mask(toy_adata):
    """Fixture to create a toy AnnData object with boolean mask for testing."""
    # Create a boolean mask - use first 15 features
    mask = np.array([True] * 15 + [False] * 5)
    toy_adata.var["feature_mask"] = mask
    return toy_adata


def test_run_pca_default(toy_adata):
    """Test the pca function with default parameters (obs space)."""
    at.tl.pca(toy_adata)

    # Check default storage locations for obs space PCA
    assert "X_pca_obs" in toy_adata.obsm, "PCA coordinates not found in obsm"
    assert "PCs_obs" in toy_adata.varm, "PCA loadings not found in varm"
    assert "variance_pca_obs" in toy_adata.uns, "PCA variance not found in uns"

    # Check shapes
    assert toy_adata.obsm["X_pca_obs"].shape[0] == toy_adata.n_obs
    assert toy_adata.varm["PCs_obs"].shape[0] == toy_adata.n_vars

    # Check variance information
    assert "variance_ratio" in toy_adata.uns["variance_pca_obs"]
    assert "variance" in toy_adata.uns["variance_pca_obs"]


def test_run_pca_var_space(toy_adata):
    """Test the pca function in var space (PCA on genes)."""
    at.tl.pca(toy_adata, dim_space="var")

    # Check storage locations for var space PCA
    assert "X_pca_var" in toy_adata.varm, "PCA coordinates not found in varm"
    assert "PCs_var" in toy_adata.obsm, "PCA loadings not found in obsm"
    assert "variance_pca_var" in toy_adata.uns, "PCA variance not found in uns"

    # Check shapes
    assert toy_adata.varm["X_pca_var"].shape[0] == toy_adata.n_vars
    assert toy_adata.obsm["PCs_var"].shape[0] == toy_adata.n_obs


def test_run_pca_with_layer(toy_adata_with_layers):
    """Test the pca function using a specific layer."""
    at.tl.pca(toy_adata_with_layers, layer="norm")

    # Check that PCA results exist
    assert "X_pca_obs" in toy_adata_with_layers.obsm
    assert "PCs_obs" in toy_adata_with_layers.varm
    assert "variance_pca_obs" in toy_adata_with_layers.uns


def test_run_pca_with_custom_embeddings_name(toy_adata):
    """Test the pca function with custom embeddings name."""
    custom_name = "my_custom_pca"
    at.tl.pca(toy_adata, embeddings_name=custom_name)

    # Check custom naming
    assert custom_name in toy_adata.obsm, f"Custom PCA coordinates not found with name {custom_name}"
    assert custom_name in toy_adata.varm, f"Custom PCA loadings not found with name {custom_name}"
    assert custom_name in toy_adata.uns, f"Custom PCA variance not found with name {custom_name}"


def test_run_pca_with_mask(toy_adata_with_mask):
    """Test the pca function with feature mask."""
    at.tl.pca(toy_adata_with_mask, meta_data_mask_column_name="feature_mask")

    # Check that PCA results exist
    assert "X_pca_obs" in toy_adata_with_mask.obsm
    assert "PCs_obs" in toy_adata_with_mask.varm
    assert "variance_pca_obs" in toy_adata_with_mask.uns

    # Check that loadings have NaN for masked features
    loadings = toy_adata_with_mask.varm["PCs_obs"]
    mask = toy_adata_with_mask.var["feature_mask"].values

    # Features not in mask should have NaN loadings
    assert np.isnan(loadings[~mask, :]).all(), "Masked features should have NaN loadings"
    # Features in mask should not have NaN loadings
    assert not np.isnan(loadings[mask, :]).any(), "Unmasked features should not have NaN loadings"


def test_run_pca_var_space_with_mask(toy_adata_with_mask):
    """Test the pca function in var space with feature mask."""
    at.tl.pca(toy_adata_with_mask, dim_space="var", meta_data_mask_column_name="feature_mask")

    # Check that PCA results exist in correct locations
    assert "X_pca_var" in toy_adata_with_mask.varm
    assert "PCs_var" in toy_adata_with_mask.obsm
    assert "variance_pca_var" in toy_adata_with_mask.uns

    # Check that coordinates have NaN for masked features
    coordinates = toy_adata_with_mask.varm["X_pca_var"]
    mask = toy_adata_with_mask.var["feature_mask"].values

    # Features not in mask should have NaN coordinates
    assert np.isnan(coordinates[~mask, :]).all(), "Masked features should have NaN coordinates"
    # Features in mask should not have NaN coordinates
    assert not np.isnan(coordinates[mask, :]).any(), "Unmasked features should not have NaN coordinates"


# Legacy test for backward compatibility
def test_run_pca_legacy(toy_adata):
    """Test the run_pca function on a toy dataset (legacy test)."""
    toy_adata.layers["norm"] = toy_adata.X.copy()
    at.tl.pca(toy_adata, layer="norm")

    # Assertions for Expected Outputs (checking default obs space)
    assert "X_pca_obs" in toy_adata.obsm, "PCA results not found in obsm"
    assert "variance_pca_obs" in toy_adata.uns, "PCA metadata not found in uns"
    assert "PCs_obs" in toy_adata.varm, "Principal components not found in varm"

    # Check for API consistency
    required_attrs = {"X_pca_obs", "variance_pca_obs", "PCs_obs"}
    existing_attrs = set(toy_adata.obsm.keys()).union(toy_adata.uns.keys(), toy_adata.varm.keys())
    missing_attrs = required_attrs - existing_attrs
    assert not missing_attrs, f"Expected attributes missing: {missing_attrs}"
