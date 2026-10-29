from typing import Any

import numpy as np
import pytest
from anndata import AnnData, read_h5ad

import alphapepttools as apt


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


def test_pca__default(toy_adata):
    """Test the pca function with default parameters."""
    apt.tl.pca(toy_adata)

    # Check default storage locations
    assert "X_pca" in toy_adata.obsm
    assert "PCs_pca" in toy_adata.varm
    assert "variance_pca" in toy_adata.uns

    # Check shapes
    assert toy_adata.obsm["X_pca"].shape[0] == toy_adata.n_obs
    assert toy_adata.varm["PCs_pca"].shape[0] == toy_adata.n_vars

    # Check variance information
    assert "variance_ratio" in toy_adata.uns["variance_pca"]
    assert "variance" in toy_adata.uns["variance_pca"]

    # Check that the fitted sample and feature names are recorded
    np.testing.assert_array_equal(toy_adata.uns["variance_pca"]["obs_names"], toy_adata.obs_names)
    np.testing.assert_array_equal(toy_adata.uns["variance_pca"]["var_names"], toy_adata.var_names)


def test_pca__copy(toy_adata) -> None:
    """Test the pca function correctly handles copy behaviour."""
    new_adata = apt.tl.pca(toy_adata, n_comps=5, copy=True)

    # Check that original data was not modified
    assert "X_pca" not in toy_adata.obsm
    assert "PCs_pca" not in toy_adata.varm
    assert "variance_pca" not in toy_adata.uns

    # Check default storage locations
    assert "X_pca" in new_adata.obsm
    assert "PCs_pca" in new_adata.varm
    assert "variance_pca" in new_adata.uns

    # Check shapes
    assert new_adata.obsm["X_pca"].shape[0] == new_adata.n_obs
    assert new_adata.varm["PCs_pca"].shape[0] == new_adata.n_vars

    # Check variance information
    assert "variance_ratio" in new_adata.uns["variance_pca"]
    assert "variance" in new_adata.uns["variance_pca"]


def test_pca__with_layer(toy_adata_with_layers):
    """Test the pca function using a specific layer."""
    apt.tl.pca(toy_adata_with_layers, layer="norm")

    # Check that PCA results exist
    assert "X_pca" in toy_adata_with_layers.obsm
    assert "PCs_pca" in toy_adata_with_layers.varm
    assert "variance_pca" in toy_adata_with_layers.uns


def test_pca__with_custom_embeddings_name(toy_adata):
    """Test the pca function with custom embeddings name."""
    custom_name = "my_custom_pca"
    apt.tl.pca(toy_adata, embeddings_name=custom_name)

    # Check custom naming
    assert custom_name in toy_adata.obsm, f"Custom PCA coordinates not found with name {custom_name}"
    assert custom_name in toy_adata.varm, f"Custom PCA loadings not found with name {custom_name}"
    assert custom_name in toy_adata.uns, f"Custom PCA variance not found with name {custom_name}"


def test_pca__with_mask(toy_adata_with_mask):
    """Test the pca function with feature mask."""
    apt.tl.pca(toy_adata_with_mask, feature_mask_column="feature_mask")

    # Check that PCA results exist
    assert "X_pca" in toy_adata_with_mask.obsm
    assert "PCs_pca" in toy_adata_with_mask.varm
    assert "variance_pca" in toy_adata_with_mask.uns

    # Check that loadings have NaN for masked features
    loadings = toy_adata_with_mask.varm["PCs_pca"]
    mask = toy_adata_with_mask.var["feature_mask"].values

    # Features not in mask should have NaN loadings
    assert np.isnan(loadings[~mask, :]).all(), "Masked features should have NaN loadings"
    # Features in mask should not have NaN loadings
    assert not np.isnan(loadings[mask, :]).any(), "Unmasked features should not have NaN loadings"

    # Only the masked-in features are recorded as fitted; all samples are
    variance = toy_adata_with_mask.uns["variance_pca"]
    np.testing.assert_array_equal(variance["var_names"], toy_adata_with_mask.var_names[mask])
    np.testing.assert_array_equal(variance["obs_names"], toy_adata_with_mask.obs_names)


def test_pca__transposed_adata(toy_adata):
    """PCA of the features is obtained by passing the transposed object."""
    adata_t = toy_adata.T.copy()
    apt.tl.pca(adata_t, n_comps=5)

    # samples and features swap roles: features are the observations of the transposed object
    assert adata_t.obsm["X_pca"].shape == (toy_adata.n_vars, 5)
    assert adata_t.varm["PCs_pca"].shape == (toy_adata.n_obs, 5)
    np.testing.assert_array_equal(adata_t.uns["variance_pca"]["obs_names"], toy_adata.var_names)
    np.testing.assert_array_equal(adata_t.uns["variance_pca"]["var_names"], toy_adata.obs_names)


def test_pca__uns_round_trips_h5ad(toy_adata_with_mask, tmp_path):
    """The recorded names must survive writing to and reading from h5ad."""
    apt.tl.pca(toy_adata_with_mask, feature_mask_column="feature_mask", n_comps=5)
    path = tmp_path / "pca.h5ad"
    toy_adata_with_mask.write_h5ad(path)

    loaded = read_h5ad(path)

    variance = loaded.uns["variance_pca"]
    np.testing.assert_array_equal(variance["obs_names"], toy_adata_with_mask.obs_names)
    np.testing.assert_array_equal(
        variance["var_names"], toy_adata_with_mask.var_names[toy_adata_with_mask.var["feature_mask"].values]
    )


def test_pca__legacy(toy_adata):
    """Test the run_pca function on a toy dataset (legacy test)."""
    toy_adata.layers["norm"] = toy_adata.X.copy()
    apt.tl.pca(toy_adata, layer="norm")

    # Assertions for Expected Outputs
    assert "X_pca" in toy_adata.obsm, "PCA results not found in obsm"
    assert "variance_pca" in toy_adata.uns, "PCA metadata not found in uns"
    assert "PCs_pca" in toy_adata.varm, "Principal components not found in varm"

    # Check for API consistency
    required_attrs = {"X_pca", "variance_pca", "PCs_pca"}
    existing_attrs = set(toy_adata.obsm.keys()).union(toy_adata.uns.keys(), toy_adata.varm.keys())
    missing_attrs = required_attrs - existing_attrs
    assert not missing_attrs, f"Expected attributes missing: {missing_attrs}"


@pytest.fixture
def toy_adata_with_missing_values() -> dict[str, Any]:
    """Fixture to create a toy AnnData object with missing values for testing."""
    n_obs, n_var, n_latent = 100, 20, 5
    missing_fraction = 0.1
    rng = np.random.default_rng(seed=42)

    usage = rng.standard_normal(size=(n_obs, n_latent))
    loadings = rng.standard_normal(size=(n_latent, n_var))
    data = usage @ loadings  # 100 samples, 20 features, 5 latent factors

    # Introduce ~10% missing values
    missing_mask = np.random.random(data.shape) < missing_fraction
    data[missing_mask] = np.nan
    var_names = [f"gene_{i}" for i in range(n_var)]
    obs_names = [f"cell_{i}" for i in range(n_obs)]
    return {
        "adata": AnnData(X=data, var={"var_names": var_names}, obs={"obs_names": obs_names}),
        "n_obs": n_obs,
        "n_var": n_var,
        "n_latent": n_latent,
    }


def test_bpca__default(toy_adata):
    """Test the bpca function with default parameters."""
    apt.tl.bpca(toy_adata, n_comps=5)

    assert "X_bpca" in toy_adata.obsm
    assert "PCs_bpca" in toy_adata.varm
    assert "variance_bpca" in toy_adata.uns

    assert toy_adata.obsm["X_bpca"].shape == (toy_adata.n_obs, 5)
    assert toy_adata.varm["PCs_bpca"].shape == (toy_adata.n_vars, 5)

    assert "variance_ratio" in toy_adata.uns["variance_bpca"]
    assert len(toy_adata.uns["variance_bpca"]["variance_ratio"]) == 5  # noqa: PLR2004

    np.testing.assert_array_equal(toy_adata.uns["variance_bpca"]["obs_names"], toy_adata.obs_names)
    np.testing.assert_array_equal(toy_adata.uns["variance_bpca"]["var_names"], toy_adata.var_names)


def test_bpca__copy(toy_adata) -> None:
    """Test the bpca function correctly handles copy behaviour."""
    new_adata = apt.tl.bpca(toy_adata, n_comps=5, copy=True)

    # Make sure that original adata was not modified
    assert "X_bpca" not in toy_adata.obsm
    assert "variance_bpca" not in toy_adata.uns

    # Make sure that new adata object contains the expected fields
    assert "X_bpca" in new_adata.obsm
    assert "PCs_bpca" in new_adata.varm
    assert "variance_bpca" in new_adata.uns

    assert new_adata.obsm["X_bpca"].shape == (new_adata.n_obs, 5)
    assert new_adata.varm["PCs_bpca"].shape == (new_adata.n_vars, 5)

    assert "variance_ratio" in new_adata.uns["variance_bpca"]
    assert len(new_adata.uns["variance_bpca"]["variance_ratio"]) == 5  # noqa: PLR2004


def test_bpca__with_layer(toy_adata_with_layers):
    """Test the bpca function using a specific layer."""
    apt.tl.bpca(toy_adata_with_layers, layer="norm", n_comps=5)

    assert "PCs_bpca" in toy_adata_with_layers.varm
    assert "X_bpca" in toy_adata_with_layers.obsm
    assert "variance_bpca" in toy_adata_with_layers.uns

    assert toy_adata_with_layers.varm["PCs_bpca"].shape == (toy_adata_with_layers.n_vars, 5)
    assert toy_adata_with_layers.obsm["X_bpca"].shape == (toy_adata_with_layers.n_obs, 5)


def test_bpca__with_mask(toy_adata_with_mask):
    """Test the bpca function with feature mask."""
    apt.tl.bpca(toy_adata_with_mask, feature_mask_column="feature_mask", n_comps=5)

    assert "X_bpca" in toy_adata_with_mask.obsm
    assert "PCs_bpca" in toy_adata_with_mask.varm
    assert "variance_bpca" in toy_adata_with_mask.uns

    # Check that loadings have NaN for masked features
    loadings = toy_adata_with_mask.varm["PCs_bpca"]
    mask = toy_adata_with_mask.var["feature_mask"].values

    # Features not in mask should have NaN loadings
    assert np.isnan(loadings[~mask, :]).all(), "Masked features should have NaN loadings"
    # Features in mask should not have NaN loadings
    assert not np.isnan(loadings[mask, :]).any(), "Unmasked features should not have NaN loadings"

    variance = toy_adata_with_mask.uns["variance_bpca"]
    np.testing.assert_array_equal(variance["var_names"], toy_adata_with_mask.var_names[mask])
    np.testing.assert_array_equal(variance["obs_names"], toy_adata_with_mask.obs_names)


def test_bpca__with_missing_values(toy_adata_with_missing_values):
    """Test the bpca function with data containing missing values."""
    adata = toy_adata_with_missing_values["adata"]
    n_obs = toy_adata_with_missing_values["n_obs"]
    n_var = toy_adata_with_missing_values["n_var"]
    n_latent = toy_adata_with_missing_values["n_latent"]

    apt.tl.bpca(adata, n_comps=n_latent)

    # Assert - Check that BPCA results exist in correct locations
    assert "X_bpca" in adata.obsm
    assert "PCs_bpca" in adata.varm
    assert "variance_bpca" in adata.uns

    # Check shapes
    assert adata.obsm["X_bpca"].shape == (n_obs, n_latent)
    assert adata.varm["PCs_bpca"].shape == (n_var, n_latent)

    # Check that BPCA output does not contain NaN values (BPCA should handle missing data)
    assert not np.isnan(adata.obsm["X_bpca"]).any()
    assert not np.isnan(adata.varm["PCs_bpca"]).any()


def test_bpca__returns_variance_ratio(toy_adata_with_missing_values):
    """Test that BPCA returns variance ratio information."""
    adata = toy_adata_with_missing_values["adata"]
    n_latent = toy_adata_with_missing_values["n_latent"]

    apt.tl.bpca(adata, n_comps=n_latent)

    variance_ratio = adata.uns["variance_bpca"]["variance_ratio"]
    assert len(variance_ratio) == n_latent, f"Should have {n_latent} variance ratio values"
    assert not np.isnan(variance_ratio).any()


def test_bpca__components_ordered_by_variance(toy_adata_with_missing_values):
    """Test that BPCA components are ordered by decreasing absolute variance explained."""
    adata = toy_adata_with_missing_values["adata"]
    n_latent = toy_adata_with_missing_values["n_latent"]

    apt.tl.bpca(adata, n_comps=n_latent)

    variance_ratio = adata.uns["variance_bpca"]["variance_ratio"]
    # Check that absolute variance ratios are in descending order
    assert np.all(np.diff(variance_ratio) <= 0)


### Test _check_inputs_for_dim_reduction validation branches ###


def test_pca__non_anndata_raises():
    """Non-AnnData input should raise TypeError."""
    with pytest.raises(TypeError, match="Data should be AnnData object"):
        apt.tl.pca(np.random.randn(10, 5))


def test_pca__missing_layer_raises(toy_adata):
    """Unknown layer name should raise ValueError."""
    with pytest.raises(ValueError, match="not found in AnnData"):
        apt.tl.pca(toy_adata, layer="nonexistent_layer")


def test_pca__missing_mask_column_raises(toy_adata):
    """Unknown `feature_mask_column` should raise ValueError."""
    with pytest.raises(ValueError, match="not found in data.var"):
        apt.tl.pca(toy_adata, feature_mask_column="nonexistent_column")


def test_pca__non_boolean_mask_column_raises(toy_adata):
    """A mask column that is not boolean should raise TypeError."""
    toy_adata.var["int_mask"] = list(range(toy_adata.n_vars))  # int, not bool
    with pytest.raises(TypeError, match="must be of boolean dtype"):
        apt.tl.pca(toy_adata, feature_mask_column="int_mask")


### Test _store_pca_results overwrite warnings ###


def test_pca__warns_on_existing_keys(toy_adata, caplog):
    """A second pca() call should warn that variance, coords, and loadings keys will be overwritten."""
    # First call populates the default keys
    apt.tl.pca(toy_adata)

    import logging

    # Second call should produce three warnings (variance, coords, loadings)
    with caplog.at_level(logging.WARNING):
        apt.tl.pca(toy_adata)

    assert "Overwriting existing PCA variance" in caplog.text
    assert "Overwriting existing PCA coordinates" in caplog.text
    assert "Overwriting existing PCA loadings" in caplog.text
