# test_plot_data_handling.py

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.tl.plot_data_handling import (
    _validate_pca_loadings_plot_inputs,
    _validate_pca_plot_input,
    _validate_scree_plot_input,
    extract_pca_anndata,
    prepare_pca_1d_loadings_data_to_plot,
    prepare_pca_2d_loadings_data_to_plot,
    prepare_scree_data_to_plot,
)


@pytest.fixture
def sample_adata():
    """Create a sample AnnData object with PCA results for testing."""
    # Create sample data
    n_obs, n_vars = 100, 50
    X = np.random.randn(n_obs, n_vars)

    # Create AnnData object
    adata = ad.AnnData(X)
    adata.obs_names = [f"sample_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]

    # Add sample metadata
    adata.obs["condition"] = np.random.choice(["A", "B", "C"], n_obs)
    adata.obs["batch"] = np.random.choice([1, 2], n_obs)
    adata.var["gene_type"] = np.random.choice(["protein_coding", "lncRNA"], n_vars)

    # Add PCA results for obs space
    n_pcs = 10
    adata.obsm["X_pca_obs"] = np.random.randn(n_obs, n_pcs)
    adata.varm["PCs_obs"] = np.random.randn(n_vars, n_pcs)
    adata.uns["variance_pca_obs"] = {"variance_ratio": np.random.rand(n_pcs), "variance": np.random.rand(n_pcs) * 100}

    # Add PCA results for var space
    adata.varm["X_pca_var"] = np.random.randn(n_vars, n_pcs)
    adata.obsm["PCs_var"] = np.random.randn(n_obs, n_pcs)
    adata.uns["variance_pca_var"] = {"variance_ratio": np.random.rand(n_pcs), "variance": np.random.rand(n_pcs) * 100}

    # Add custom embeddings
    adata.obsm["custom_embedding"] = np.random.randn(n_obs, 5)
    adata.varm["custom_loadings"] = np.random.randn(n_vars, 5)
    adata.uns["custom_embedding"] = {"variance_ratio": np.random.rand(5), "variance": np.random.rand(5) * 100}

    return adata


class TestValidationFunctions:
    """Test validation functions for plot data handling."""

    def test_validate_pca_plot_input_valid(self, sample_adata):
        """Test _validate_pca_plot_input with valid inputs."""
        # Should not raise any errors
        _validate_pca_plot_input(sample_adata, "X_pca_obs", "variance_pca_obs", "obs")
        _validate_pca_plot_input(sample_adata, "X_pca_var", "variance_pca_var", "var")
        _validate_pca_plot_input(sample_adata, "custom_embedding", "custom_embedding", "obs")

    def test_validate_pca_plot_input_invalid_data_type(self):
        """Test _validate_pca_plot_input with invalid data type."""
        with pytest.raises(TypeError, match="data must be an AnnData object"):
            _validate_pca_plot_input("not_anndata", "X_pca_obs", "variance_pca_obs", "obs")

    def test_validate_pca_plot_input_invalid_dim_space(self, sample_adata):
        """Test _validate_pca_plot_input with invalid dim_space."""
        with pytest.raises(ValueError, match="dim_space must be either 'obs' or 'var'"):
            _validate_pca_plot_input(sample_adata, "X_pca_obs", "variance_pca_obs", "invalid")

    def test_validate_pca_plot_input_missing_layer(self, sample_adata):
        """Test _validate_pca_plot_input with missing PCA layer."""
        with pytest.raises(ValueError, match="PCA embeddings layer 'missing_layer' not found"):
            _validate_pca_plot_input(sample_adata, "missing_layer", "variance_pca_obs", "obs")

    def test_validate_scree_plot_input_valid(self, sample_adata):
        """Test _validate_scree_plot_input with valid inputs."""
        # Should not raise any errors
        _validate_scree_plot_input(sample_adata, 5, "obs", "variance_pca_obs")
        _validate_scree_plot_input(sample_adata, 10, "var", "variance_pca_var")

    def test_validate_scree_plot_input_invalid_data_type(self):
        """Test _validate_scree_plot_input with invalid data type."""
        with pytest.raises(TypeError, match="data must be an AnnData object"):
            _validate_scree_plot_input("not_anndata", 5, "obs", "variance_pca_obs")

    def test_validate_scree_plot_input_invalid_dim_space(self, sample_adata):
        """Test _validate_scree_plot_input with invalid dim_space."""
        with pytest.raises(ValueError, match="dim_space must be either 'obs' or 'var'"):
            _validate_scree_plot_input(sample_adata, 5, "invalid", "variance_pca_obs")

    def test_validate_scree_plot_input_missing_variance_layer(self, sample_adata):
        """Test _validate_scree_plot_input with missing variance layer."""
        with pytest.raises(ValueError, match="PCA metadata layer 'missing_variance' not found"):
            _validate_scree_plot_input(sample_adata, 5, "obs", "missing_variance")

    def test_validate_scree_plot_input_too_many_pcs(self, sample_adata, caplog):
        """Test _validate_scree_plot_input with too many PCs requested."""
        _validate_scree_plot_input(sample_adata, 20, "obs", "variance_pca_obs")
        assert "Requested 20 PCs, but only 10 PCs are available" in caplog.text

    def test_validate_pca_loadings_plot_inputs_valid(self, sample_adata):
        """Test _validate_pca_loadings_plot_inputs with valid inputs."""
        # Should not raise any errors
        _validate_pca_loadings_plot_inputs(sample_adata, "PCs_obs", 1, 2, 10, "obs")
        _validate_pca_loadings_plot_inputs(sample_adata, "PCs_var", 1, None, 5, "var")

    def test_validate_pca_loadings_plot_inputs_invalid_data_type(self):
        """Test _validate_pca_loadings_plot_inputs with invalid data type."""
        with pytest.raises(TypeError, match="data must be an AnnData object"):
            _validate_pca_loadings_plot_inputs("not_anndata", "PCs_obs", 1, 2, 10, "obs")

    def test_validate_pca_loadings_plot_inputs_invalid_dim_space(self, sample_adata):
        """Test _validate_pca_loadings_plot_inputs with invalid dim_space."""
        with pytest.raises(ValueError, match="dim_space must be either 'obs' or 'var'"):
            _validate_pca_loadings_plot_inputs(sample_adata, "PCs_obs", 1, 2, 10, "invalid")

    def test_validate_pca_loadings_plot_inputs_missing_loadings(self, sample_adata):
        """Test _validate_pca_loadings_plot_inputs with missing loadings layer."""
        with pytest.raises(ValueError, match="PCA feature loadings layer 'missing_loadings' not found"):
            _validate_pca_loadings_plot_inputs(sample_adata, "missing_loadings", 1, 2, 10, "obs")

    def test_validate_pca_loadings_plot_inputs_invalid_pc(self, sample_adata):
        """Test _validate_pca_loadings_plot_inputs with invalid PC dimensions."""
        with pytest.raises(ValueError, match="PC must be between 1 and 10"):
            _validate_pca_loadings_plot_inputs(sample_adata, "PCs_obs", 0, 2, 10, "obs")

        with pytest.raises(ValueError, match="second PC must be between 1 and 10"):
            _validate_pca_loadings_plot_inputs(sample_adata, "PCs_obs", 1, 15, 10, "obs")

    def test_validate_pca_loadings_plot_inputs_invalid_nfeatures(self, sample_adata):
        """Test _validate_pca_loadings_plot_inputs with invalid number of features."""
        with pytest.raises(ValueError, match="Number of features must be between 1 and 50"):
            _validate_pca_loadings_plot_inputs(sample_adata, "PCs_obs", 1, 2, 0, "obs")

        with pytest.raises(ValueError, match="Number of features must be between 1 and 50"):
            _validate_pca_loadings_plot_inputs(sample_adata, "PCs_obs", 1, 2, 100, "obs")


class TestDataPreparationFunctions:
    def test_prepare_scree_data_to_plot_basic(self, sample_adata):
        """Test prepare_scree_data_to_plot with basic parameters."""
        result = prepare_scree_data_to_plot(sample_adata, n_pcs=5, dim_space="obs")

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["PC", "explained_variance", "explained_variance_percent"]
        assert list(result["PC"]) == [1, 2, 3, 4, 5]
        assert all(0 <= x <= 1 for x in result["explained_variance"])

    def test_prepare_scree_data_to_plot_var_space(self, sample_adata):
        """Test prepare_scree_data_to_plot with var space."""
        result = prepare_scree_data_to_plot(sample_adata, n_pcs=3, dim_space="var")

        assert isinstance(result, pd.DataFrame)

    def test_prepare_scree_data_to_plot_custom_embedding(self, sample_adata):
        """Test prepare_scree_data_to_plot with custom embedding name."""
        result = prepare_scree_data_to_plot(sample_adata, n_pcs=3, dim_space="obs", embeddings_name="custom_embedding")

        assert isinstance(result, pd.DataFrame)

    def test_prepare_scree_data_to_plot_too_many_pcs(self, sample_adata):
        """Test prepare_scree_data_to_plot with more PCs than available."""
        result = prepare_scree_data_to_plot(sample_adata, n_pcs=20, dim_space="obs")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10  # Should be limited to available PCs  # noqa: PLR2004

    def test_prepare_pca_1d_loadings_data_to_plot_basic(self, sample_adata):
        """Test prepare_pca_1d_loadings_data_to_plot with basic parameters."""
        result = prepare_pca_1d_loadings_data_to_plot(sample_adata, dim_space="obs", dim=1, nfeatures=10)

        assert isinstance(result, pd.DataFrame)
        expected_columns = ["dim_loadings", "feature", "abs_loadings", "index_int"]
        assert list(result.columns) == expected_columns
        assert list(result["index_int"]) == list(range(10, 0, -1))
        assert result["feature"].dtype == "string"

    def test_prepare_pca_1d_loadings_data_to_plot_var_space(self, sample_adata):
        """Test prepare_pca_1d_loadings_data_to_plot with var space."""
        result = prepare_pca_1d_loadings_data_to_plot(sample_adata, dim_space="var", dim=1, nfeatures=5)

        assert isinstance(result, pd.DataFrame)
        # In var space, features should be sample names
        assert all(feature.startswith("sample_") for feature in result["feature"])

    def test_prepare_pca_1d_loadings_data_to_plot_custom_embedding(self, sample_adata):
        """Test prepare_pca_1d_loadings_data_to_plot with custom embedding name."""
        result = prepare_pca_1d_loadings_data_to_plot(
            sample_adata, dim_space="obs", dim=1, nfeatures=5, embeddings_name="custom_loadings"
        )

        assert isinstance(result, pd.DataFrame)

    def test_prepare_pca_2d_loadings_data_to_plot_basic(self, sample_adata):
        """Test prepare_pca_2d_loadings_data_to_plot with basic parameters."""
        result = prepare_pca_2d_loadings_data_to_plot(
            sample_adata, loadings_name="PCs_obs", pc_x=1, pc_y=2, nfeatures=10, dim_space="obs"
        )

        assert isinstance(result, pd.DataFrame)
        expected_columns = ["dim1_loadings", "dim2_loadings", "feature", "abs_dim1", "abs_dim2", "is_top"]
        assert list(result.columns) == expected_columns
        assert len(result) <= 50  # Should be filtered for non-NaN features  # noqa: PLR2004
        assert "is_top" in result.columns
        assert result["is_top"].dtype == bool

    def test_prepare_pca_2d_loadings_data_to_plot_var_space(self, sample_adata):
        """Test prepare_pca_2d_loadings_data_to_plot with var space."""
        result = prepare_pca_2d_loadings_data_to_plot(
            sample_adata, loadings_name="PCs_var", pc_x=1, pc_y=2, nfeatures=5, dim_space="var"
        )

        assert isinstance(result, pd.DataFrame)
        # In var space, features should be sample names
        assert all(feature.startswith("sample_") for feature in result["feature"])


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_prepare_pca_data_with_nan_values(self):
        """Test data preparation functions with NaN values in data."""
        # Create AnnData with some NaN values
        n_obs, n_vars = 10, 5
        X = np.random.randn(n_obs, n_vars)
        X[0, 0] = np.nan

        adata = ad.AnnData(X)
        adata.obs_names = [f"sample_{i}" for i in range(n_obs)]
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs["condition"] = ["A"] * 5 + ["B"] * 5

        # Add PCA results with some NaN loadings
        n_pcs = 3
        adata.obsm["X_pca_obs"] = np.random.randn(n_obs, n_pcs)
        loadings = np.random.randn(n_vars, n_pcs)
        loadings[0, :] = np.nan  # Make first gene have NaN loadings
        adata.varm["PCs_obs"] = loadings
        adata.uns["variance_pca_obs"] = {
            "variance_ratio": np.random.rand(n_pcs),
            "variance": np.random.rand(n_pcs) * 100,
        }

        # Test loadings with NaN values
        result = prepare_pca_2d_loadings_data_to_plot(adata, "PCs_obs", 1, 2, 3, "obs")
        # Should filter out features with all-NaN loadings
        assert len(result) == 4  # 5 features - 1 with all NaN  # noqa: PLR2004

    def test_small_dataset_edge_cases(self):
        """Test with very small datasets."""
        # Create minimal AnnData
        n_obs, n_vars = 3, 2
        X = np.random.randn(n_obs, n_vars)

        adata = ad.AnnData(X)
        adata.obs_names = [f"sample_{i}" for i in range(n_obs)]
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]

        n_pcs = 2
        adata.obsm["X_pca_obs"] = np.random.randn(n_obs, n_pcs)
        adata.varm["PCs_obs"] = np.random.randn(n_vars, n_pcs)
        adata.uns["variance_pca_obs"] = {
            "variance_ratio": np.random.rand(n_pcs),
            "variance": np.random.rand(n_pcs) * 100,
        }

    def test_single_pc_case(self):
        """Test edge case with only one PC."""
        n_obs, n_vars = 10, 5
        X = np.random.randn(n_obs, n_vars)

        adata = ad.AnnData(X)
        adata.obs_names = [f"sample_{i}" for i in range(n_obs)]
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]

        # Only one PC
        n_pcs = 1
        adata.obsm["X_pca_obs"] = np.random.randn(n_obs, n_pcs)
        adata.varm["PCs_obs"] = np.random.randn(n_vars, n_pcs)
        adata.uns["variance_pca_obs"] = {
            "variance_ratio": np.random.rand(n_pcs),
            "variance": np.random.rand(n_pcs) * 100,
        }

        # Should work for single PC
        result = prepare_pca_1d_loadings_data_to_plot(adata, "obs", 1, 3)
        assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize("nfeatures", [1, 5, 10, 20])
def test_parametrized_loadings_nfeatures(sample_adata, nfeatures):
    """Parametrized test for different numbers of features in loadings plots."""
    result = prepare_pca_1d_loadings_data_to_plot(sample_adata, dim_space="obs", dim=1, nfeatures=nfeatures)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == min(nfeatures, 50)  # Limited by available features
    assert list(result["index_int"]) == list(range(min(nfeatures, 50), 0, -1))


def make_basic_anndata(n_obs=100, n_vars=200, n_pcs=2):
    X = np.arange(n_obs * n_vars, dtype=float).reshape(n_obs, n_vars)
    obs = pd.DataFrame({"sample": [f"s{i}" for i in range(n_obs)]}, index=[f"s{i}" for i in range(n_obs)])
    var = pd.DataFrame({"feature": [f"g{i}" for i in range(n_vars)]}, index=[f"g{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    # default PCA embeddings for obs and var
    adata.obsm["X_pca_obs"] = np.linspace(0, 1, n_obs * n_pcs).reshape(n_obs, n_pcs)
    adata.varm["X_pca_var"] = np.linspace(0, 1, n_vars * n_pcs).reshape(n_vars, n_pcs)
    # variance metadata
    adata.uns["variance_pca_obs"] = {"variance_ratio": np.array([0.7, 0.3])}
    adata.uns["variance_pca_var"] = {"variance_ratio": np.array([0.5, 0.5])}

    # costom embeddings names
    adata.obsm["custom_emb"] = np.ones((n_obs, n_pcs))
    adata.uns["custom_emb"] = {"variance_ratio": np.array([0.6, 0.4])}
    return adata


def test_extract_pca_anndata_obs_basic():
    adata = make_basic_anndata(n_obs=100, n_vars=200, n_pcs=2)
    out = extract_pca_anndata(adata, dim_space="obs")
    # X should be the obs PCA coordinates
    assert out.X.shape == (adata.n_obs, 2)
    # obs should be same as original obs
    assert out.obs.reset_index(drop=True).equals(adata.obs.reset_index(drop=True))
    # var should be a DataFrame built from uns variance metadata
    assert "variance_ratio" in out.var.columns
    # var_names should be stringified PC indices
    assert out.var_names.tolist() == ["1", "2"]


def test_extract_pca_anndata_var_basic():
    adata = make_basic_anndata(n_obs=5, n_vars=3, n_pcs=2)
    out = extract_pca_anndata(adata, dim_space="var")
    # X should be the var PCA coordinates (n_vars x n_pcs)
    assert out.X.shape == (adata.n_vars, 2)
    # obs in returned AnnData should equal original var (since dim_space='var')
    assert out.obs.reset_index(drop=True).equals(adata.var.reset_index(drop=True))
    # var should contain variance metadata produced from uns
    assert "variance_ratio" in out.var.columns
    assert out.var_names.tolist() == ["1", "2"]


def test_extract_pca_anndata_with_expression_columns_obs():
    adata = make_basic_anndata(n_obs=4, n_vars=4, n_pcs=2)
    # choose a subset of var_names to attach as expression columns
    expr_cols = [adata.var_names[0], adata.var_names[2]]
    out = extract_pca_anndata(adata, dim_space="obs", expression_columns=expr_cols)
    # returned.obs should include the expression columns
    for c in expr_cols:
        assert c in out.obs.columns
        # values should be numeric
        assert np.issubdtype(out.obs[c].dtype, np.number)


def test_extract_pca_anndata_missing_embeddings_raises():
    adata = make_basic_anndata()
    # remove embedding to trigger validation error
    adata.obsm.pop("X_pca_obs", None)
    with pytest.raises(ValueError):
        extract_pca_anndata(adata, dim_space="obs")


def test_extract_pca_anndata_custom_embeddings_name():
    adata = make_basic_anndata(n_obs=3, n_vars=3, n_pcs=2)
    out = extract_pca_anndata(adata, dim_space="obs", embeddings_name="custom_emb")
    assert out.X.shape == (adata.n_obs, 2)
    assert out.var_names.tolist() == ["1", "2"]
