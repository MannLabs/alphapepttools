"""
Auxiliary functions for handling data and formatting for PCA plot input.

These functions extract PCA coordinates, explained variance, and loadings,
and organize them into DataFrames for use in scatter plotting.

"""

import logging
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

# logging configuration
logging.basicConfig(level=logging.INFO)

## Helper function to validate plots inputs


def _validate_adata(adata: ad.AnnData) -> None:
    """Validate that data is an AnnData object.

    Parameters
    ----------
    adata
        The object to check for AnnData type.

    Raises
    ------
    TypeError
        If adata is not an AnnData object.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("data must be an AnnData object")


def _validate_pca_plot_input(
    adata: ad.AnnData,
    pca_embeddings_layer_name: str,
    pca_var_key: str,
) -> None:
    """Validates the AnnData object for PCA-related data and dimensions.

    Parameters
    ----------
    adata
        AnnData object to be validated.
    pca_embeddings_layer_name
        Name of the PCA layer to be checked, stored in `data.obsm`.
    pca_var_key
        Name of the PCA variance metadata layer to be checked, stored in `data.uns`.
    """
    _validate_adata(adata=adata)

    # Check if the PCA embeddings layer exists in obsm
    if pca_embeddings_layer_name not in adata.obsm:
        available_layers = list(adata.obsm.keys())
        raise ValueError(
            f"PCA embeddings layer '{pca_embeddings_layer_name}' not found in data.obsm. "
            f"Found layers: {available_layers}"
        )

    # Check if the variance layer exists in uns
    if pca_var_key not in adata.uns:
        raise ValueError(
            f"PCA metadata layer '{pca_var_key}' not found in AnnData object. Found layers: {list(adata.uns.keys())}"
        )


def _validate_scree_plot_input(
    adata: ad.AnnData,
    n_pcs: int,
    pca_variance_layer_name: str,
) -> None:
    """Validate inputs for scree plot of the PCA dimension.

    Parameters
    ----------
    adata
        The AnnData object containing PCA results.
    n_pcs
        The number of principal components requested for plotting.
    pca_variance_layer_name
        The name of the PCA layer (used to construct the embedding key as `data.uns[pca_name]`).
    """
    _validate_adata(adata=adata)

    if pca_variance_layer_name not in adata.uns:
        raise ValueError(
            f"PCA metadata layer '{pca_variance_layer_name}' not found in AnnData object. "
            f"Found layers: {list(adata.uns.keys())}"
        )

    n_pcs_avail = len(adata.uns[pca_variance_layer_name]["variance_ratio"])
    if n_pcs > n_pcs_avail:
        logging.warning(
            f"Requested {n_pcs} PCs, but only {n_pcs_avail} PCs are available. Plotting only the available PCs"
        )


def _validate_pca_loadings_plot_inputs(
    adata: ad.AnnData, loadings_name: str, dim: int, dim2: int | None, nfeatures: int
) -> None:
    """Validate inputs for accessing PCA feature loadings from an AnnData object.

    Parameters
    ----------
    adata
        The AnnData object containing PCA loadings data.
    loadings_name
        The key in `adata.varm` that stores PCA feature loadings (e.g., "PCs_pca").
    dim
        The principal component index (1-based) to extract loadings for.
    dim2
        The second principal component index (1-based) to extract loadings for, if applicable.
    nfeatures
        The number of top features to consider for the given component.
    """
    _validate_adata(adata=adata)

    # Check if the loadings layer exists in varm
    if loadings_name not in adata.varm:
        available_layers = list(adata.varm.keys())
        raise ValueError(
            f"PCA feature loadings layer '{loadings_name}' not found in adata.varm. Found layers: {available_layers}"
        )

    # Check PC dimensions
    n_pcs = adata.varm[loadings_name].shape[1]
    if not (1 <= dim <= n_pcs):
        raise ValueError(f"PC must be between 1 and {n_pcs} (inclusive). Got {dim=}")
    if dim2 is not None and not (1 <= dim2 <= n_pcs):
        raise ValueError(f"second PC must be between 1 and {n_pcs} (inclusive). Got pc_y={dim2}")

    # Check number of features
    n_features = adata.varm[loadings_name].shape[0]
    if not (1 <= nfeatures <= n_features):
        raise ValueError(f"Number of features must be between 1 and {n_features} (inclusive). Got {nfeatures=}")


## Functions to prepare data frames for plotting using the scatter method


def _extract_expression_df(
    adata: ad.AnnData,
    names: list[str] | str,
) -> pd.DataFrame:
    """Extract expression data from an AnnData object as a numeric DataFrame.

    Parameters
    ----------
    adata
        Source AnnData object.
    names
        Variable names to extract.

    Returns
    -------
    pd.DataFrame
        Numeric expression DataFrame with proper index and columns (obs x selected var_names).
    """
    # Normalize input to a list
    if isinstance(names, str):
        names = [names]

    # Keep only names that exist in adata.var_names
    valid_names = [n for n in names if n in adata.var_names]

    # Return empty DataFrame if nothing matches
    if not valid_names:
        return pd.DataFrame()

    # Extract the data
    expr = pd.DataFrame(adata[:, valid_names].X)
    expr.index = adata.obs_names
    expr.columns = valid_names

    # Ensure numeric
    return expr.apply(pd.to_numeric, errors="coerce")


def extract_pca_anndata(
    adata: ad.AnnData,
    embeddings_name: str | None = None,
    method: Literal["pca", "bpca"] = "pca",
    expression_columns: list[str] | None = None,
) -> ad.AnnData:
    """Extract PCA/BPCA data required for plotting from an AnnData object.

    Parameters
    ----------
    adata
        AnnData object containing PCA/BPCA results.
    embeddings_name
        Custom embeddings name or None to use the default naming scheme.
    method
        The method used for dimensionality reduction. Options are "pca" or "bpca" with "pca" as the default.
        This is used to construct the default keys if `embeddings_name` is None.
    expression_columns
        List of `var_names` to include as additional numerical column(s) in
        the returned AnnData's `.obs` for coloring PCA plots by expression.

    Returns
    -------
    ad.AnnData
        An AnnData object containing the PCA results.
        - `.X` stores the PCA embeddings, shape (observations x components)
        - `.var` contains the PCA variance information
        - `.obs` contains the corresponding metadata, and, if specified,
          additional expression values for coloring plots.
        - PCA dimensions in `.var_names` are named as `pc_1`, `pc_2`, etc.

    Examples
    --------
    Extract PCA projections after running PCA:

    .. code-block:: python

        import anndata as ad
        import pandas as pd
        import numpy as np
        import alphapepttools as at

        # Create a 5x5 dataset where 4 proteins are core (no missing values)
        X = np.array(
            [
                [10.5, 12.3, 11.8, 9.2, np.nan],  # Sample 1
                [11.2, 13.1, 12.5, 10.1, 7.5],  # Sample 2
                [9.8, 11.9, 10.2, 8.9, np.nan],  # Sample 3
                [12.1, 14.2, 13.3, 11.3, 8.2],  # Sample 4
                [10.9, 12.7, 11.5, 9.8, np.nan],  # Sample 5
            ]
        )

        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame({"sample": ["S1", "S2", "S3", "S4", "S5"], "condition": ["A", "B", "A", "B", "A"]}),
            var=pd.DataFrame({"protein": ["P1", "P2", "P3", "P4", "P5"], "is_core": [True, True, True, True, False]}),
        )

        # First run PCA on the samples
        at.tl.pca(adata, meta_data_mask_column_name="is_core", n_comps=2)

        # Extract PCA data for plotting/analysis
        pca_adata = at.tl.extract_pca_anndata(adata)
        display(pca_adata.to_df())  # DataFrame with PC1 and PC2 coordinates for each sample

        # The PCA projections are now in pca_adata.X (5 samples x 2 PCs)
        print(pca_adata.X.shape)  # (5, 2)
        print(pca_adata.var_names.tolist())  # ['pc_1', 'pc_2']

        # Access PC1 and PC2 coordinates for all samples
        pc1_coords = pca_adata[:, "pc_1"].X.flatten()
        pc2_coords = pca_adata[:, "pc_2"].X.flatten()

        # The original metadata is preserved in pca_adata.obs
        print(pca_adata.obs["condition"])  # ['A', 'B', 'A', 'B', 'A']

        # Variance explained is in pca_adata.var
        print(pca_adata.var["variance_ratio"])  # Proportion of variance per PC

    Include expression values for plotting:

    .. code-block:: python

        # Include protein expression for coloring
        pca_adata = at.tl.extract_pca_anndata(
            adata,
            expression_columns=["P1", "P2"],  # Include P1 and P2 expression values
        )

        # Now pca_adata.obs contains the original metadata plus expression values
        print(pca_adata.obs.columns)  # Contains 'sample', 'condition', 'P1', 'P2'

        # This allows coloring PCA plots by protein expression
        p1_expression = pca_adata.obs["P1"]  # Expression of protein P1 across samples

    """
    # Resolve PCA keys
    pca_coors_key = f"X_{method}" if embeddings_name is None else embeddings_name
    pca_var_key = f"variance_{method}" if embeddings_name is None else embeddings_name

    # Validate inputs
    _validate_pca_plot_input(adata=adata, pca_embeddings_layer_name=pca_coors_key, pca_var_key=pca_var_key)

    # Select PCA coordinates and metadata
    pca_coordinates = adata.obsm[pca_coors_key]
    obs_df = adata.obs

    # Add expression columns if provided
    if expression_columns is not None:
        expr_data = _extract_expression_df(
            adata=adata,
            names=expression_columns,
        )
        obs_df = obs_df.join(expr_data)

    # the uns entry also holds the fitted obs/var names, which are not per-component
    pca_variance = adata.uns[pca_var_key]
    var_df = pd.DataFrame({"variance_ratio": pca_variance["variance_ratio"], "variance": pca_variance["variance"]})

    # Initialize PCA AnnData
    adata_pca = ad.AnnData(X=pca_coordinates)
    adata_pca.obs = obs_df.copy()
    adata_pca.var = var_df.copy()

    # Name PCA dimensions
    adata_pca.var_names = [f"pc_{i + 1}" for i in range(adata_pca.X.shape[1])]

    return adata_pca


def prepare_scree_data_to_plot(
    adata: ad.AnnData,
    n_pcs: int,
    embeddings_name: str | None = None,
    method: Literal["pca", "bpca"] = "pca",
) -> pd.DataFrame:
    """Prepare scree plot data from AnnData object.

    Parameters
    ----------
    adata
        AnnData object containing PCA results.
    n_pcs
        Number of principal components to include.
    embeddings_name
        Custom embeddings name or None for default.
    method
        The method used for dimensionality reduction. Options are "pca" or "bpca" with "pca" as the default.
        This is used to construct the default keys if `embeddings_name` is None.

    Returns
    -------
    pd.DataFrame
        DataFrame with PC numbers and explained variance values.

    Examples
    --------
    Prepare data for a scree plot after running PCA:

    .. code-block:: python

        import anndata as ad
        import pandas as pd
        import numpy as np
        import alphapepttools as at

        # Create a 5x5 dataset where 4 proteins are core (no missing values)
        X = np.array(
            [
                [10.5, 12.3, 11.8, 9.2, np.nan],  # Sample 1
                [11.2, 13.1, 12.5, 10.1, 7.5],  # Sample 2
                [9.8, 11.9, 10.2, 8.9, np.nan],  # Sample 3
                [12.1, 14.2, 13.3, 11.3, 8.2],  # Sample 4
                [10.9, 12.7, 11.5, 9.8, np.nan],  # Sample 5
            ]
        )

        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame({"sample": ["S1", "S2", "S3", "S4", "S5"]}),
            var=pd.DataFrame({"protein": ["P1", "P2", "P3", "P4", "P5"], "is_core": [True, True, True, True, False]}),
        )

        # Run PCA on the samples
        at.tl.pca(adata, meta_data_mask_column_name="is_core", n_comps=2)

        # Prepare scree plot data
        scree_data = at.tl.prepare_scree_data_to_plot(adata, n_pcs=2)
        display(scree_data)

        # DataFrame contains:
        # - PC: Principal component number (1, 2)
        # - explained_variance: Proportion of variance explained (0-1)
        # - explained_variance_percent: Variance explained as percentage (0-100)

    """
    # Generate the correct variance key name
    variance_key = f"variance_{method}" if embeddings_name is None else embeddings_name

    # Input checks
    _validate_scree_plot_input(adata=adata, n_pcs=n_pcs, pca_variance_layer_name=variance_key)

    n_pcs_avail = len(adata.uns[variance_key]["variance_ratio"])
    n_pcs = min(n_pcs, n_pcs_avail)
    # Create the dataframe for plotting, X = pcs, y = explained variance

    return pd.DataFrame(
        {
            "PC": np.arange(n_pcs) + 1,
            "explained_variance": adata.uns[variance_key]["variance_ratio"][:n_pcs],
            # add the explained variance in percent format
            "explained_variance_percent": adata.uns[variance_key]["variance_ratio"][:n_pcs] * 100,
        }
    )


def prepare_pca_1d_loadings_data_to_plot(
    data: ad.AnnData,
    dim: int,
    nfeatures: int,
    embeddings_name: str | None = None,
    method: Literal["pca", "bpca"] = "pca",
) -> pd.DataFrame:
    """Prepare the gene loadings (1d) of a PC for plotting.

    Parameters
    ----------
    data
        AnnData to plot.
    dim
        The PC number from which to get loadings (1-indexed, i.e. the first PC is 1, not 0).
    nfeatures
        The number of top absolute loadings features to plot.
    embeddings_name
        The custom embeddings name used in PCA. If None, uses default naming convention.
    method
        The method used for dimensionality reduction. Options are "pca" or "bpca" with "pca" as the default.
        This is used to construct the default keys if `embeddings_name` is None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the top nfeatures loadings for the specified PC dimension.

    Examples
    --------
    Get top contributing features for a principal component:

    .. code-block:: python

        import anndata as ad
        import pandas as pd
        import numpy as np
        import alphapepttools as at

        # Create a 5x5 dataset where 4 proteins are core (no missing values)
        X = np.array(
            [
                [10.5, 12.3, 11.8, 9.2, np.nan],  # Sample 1
                [11.2, 13.1, 12.5, 10.1, 7.5],  # Sample 2
                [9.8, 11.9, 10.2, 8.9, np.nan],  # Sample 3
                [12.1, 14.2, 13.3, 11.3, 8.2],  # Sample 4
                [10.9, 12.7, 11.5, 9.8, np.nan],  # Sample 5
            ]
        )

        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame({"sample": ["S1", "S2", "S3", "S4", "S5"]}),
            var=pd.DataFrame({"protein": ["P1", "P2", "P3", "P4", "P5"], "is_core": [True, True, True, True, False]}),
        )

        # Run PCA on the samples
        at.tl.pca(adata, meta_data_mask_column_name="is_core", n_comps=2)

        # Get top 3 protein loadings for PC1
        loadings_df = at.tl.prepare_pca_1d_loadings_data_to_plot(
            adata,
            dim=1,  # PC1
            nfeatures=3,  # Top 3 proteins
        )
        display(loadings_df)

        # DataFrame contains:
        # - feature: Protein names (P1, P2, P3, P4)
        # - dim_loadings: Loading values for PC1
        # - abs_loadings: Absolute loading values
        # - index_int: Ranking index for plotting

    """
    # Generate the correct loadings key name
    loadings_key = f"PCs_{method}" if embeddings_name is None else embeddings_name

    _validate_pca_loadings_plot_inputs(adata=data, loadings_name=loadings_key, dim=dim, dim2=None, nfeatures=nfeatures)

    # create the dataframe for plotting
    dim_z = dim - 1  # to account from 0 indexing
    loadings_matrix = data.varm[loadings_key]
    loadings_df = pd.DataFrame({"dim_loadings": loadings_matrix[:, dim_z]})
    loadings_df["feature"] = data.var.index.astype("string")

    loadings_df["abs_loadings"] = loadings_df["dim_loadings"].abs()
    # Sort the DataFrame by absolute loadings and select the top features
    top_loadings_df = loadings_df.sort_values(by="abs_loadings", ascending=False).copy().head(nfeatures)
    top_loadings_df = top_loadings_df.reset_index(drop=True)
    top_loadings_df["index_int"] = range(nfeatures, 0, -1)

    return top_loadings_df


def prepare_pca_2d_loadings_data_to_plot(
    data: ad.AnnData,
    pc_x: int,
    pc_y: int,
    nfeatures: int,
    embeddings_name: str | None = None,
    method: Literal["pca", "bpca"] = "pca",
) -> pd.DataFrame:
    """Prepare a DataFrame with PCA feature loadings for the 2D plotting.

    This function extracts the loadings of two specified principal components (PCs) from
    an AnnData object, filters features that contributed to the PCA (non-zero loadings),
    and flags the top nfeatures for each selected PC dimension.

    Parameters
    ----------
    data
        The AnnData object containing PCA results.
    pc_x
        The first principal component index (1-based) to extract loadings for.
    pc_y
        The second principal component index (1-based) to extract loadings for.
    nfeatures
        Number of top features per PC to highlight based on absolute loadings.
    embeddings_name
        The custom embeddings name used in PCA. If None, uses default naming convention.
    method
        The method used for dimensionality reduction. Options are "pca" or "bpca" with "pca" as the default.
        This is used to construct the default keys if `embeddings_name` is None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing loadings for the selected PCs, feature names, boolean columns
        indicating if a feature was used in PCA and whether it is among the top features in either dimension.

    Examples
    --------
    Prepare 2D loadings data for biplot visualization:

    .. code-block:: python

        import anndata as ad
        import pandas as pd
        import numpy as np
        import alphapepttools as at

        # Create a 5x5 dataset where 4 proteins are core (no missing values)
        X = np.array(
            [
                [10.5, 12.3, 11.8, 9.2, np.nan],  # Sample 1
                [11.2, 13.1, 12.5, 10.1, 7.5],  # Sample 2
                [9.8, 11.9, 10.2, 8.9, np.nan],  # Sample 3
                [12.1, 14.2, 13.3, 11.3, 8.2],  # Sample 4
                [10.9, 12.7, 11.5, 9.8, np.nan],  # Sample 5
            ]
        )

        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame({"sample": ["S1", "S2", "S3", "S4", "S5"]}),
            var=pd.DataFrame({"protein": ["P1", "P2", "P3", "P4", "P5"], "is_core": [True, True, True, True, False]}),
        )

        # Run PCA on the samples
        at.tl.pca(adata, meta_data_mask_column_name="is_core", n_comps=2)

        # Get loadings for PC1 vs PC2 with top 2 features highlighted
        loadings_2d = at.tl.prepare_pca_2d_loadings_data_to_plot(
            adata,
            pc_x=1,  # PC1
            pc_y=2,  # PC2
            nfeatures=2,  # Top 2 features per PC
        )
        display(loadings_2d)

        # DataFrame contains:
        # - feature: Protein names (only P1-P4, P5 excluded as not core)
        # - dim1_loadings: Loading values for PC1
        # - dim2_loadings: Loading values for PC2
        # - abs_dim1, abs_dim2: Absolute loading values
        # - is_top: Boolean flag for top features in either dimension

    """
    loadings_key = f"PCs_{method}" if embeddings_name is None else embeddings_name

    _validate_pca_loadings_plot_inputs(adata=data, loadings_name=loadings_key, dim=pc_x, dim2=pc_y, nfeatures=nfeatures)

    dim1_z = pc_x - 1  # convert to 0-based index
    dim2_z = pc_y - 1  # convert to 0-based index

    orig_loadings = data.varm[loadings_key]

    loadings = pd.DataFrame(
        {
            "dim1_loadings": orig_loadings[:, dim1_z],
            "dim2_loadings": orig_loadings[:, dim2_z],
        }
    )
    loadings["feature"] = data.var_names

    # get only features that were used in the PCA (e.g., those that are part of the core proteome)
    # these would be features with all-NaN loadings in all PC dimensions
    non_nan_mask = ~np.isnan(orig_loadings).all(axis=1)
    loadings = loadings[non_nan_mask]

    # add the top N features for each dimension
    loadings["abs_dim1"] = loadings["dim1_loadings"].abs()
    loadings["abs_dim2"] = loadings["dim2_loadings"].abs()

    loadings["is_top"] = False
    loadings.loc[loadings.nlargest(nfeatures, "abs_dim1").index, "is_top"] = True
    loadings.loc[loadings.nlargest(nfeatures, "abs_dim2").index, "is_top"] = True

    return loadings
