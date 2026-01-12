"""
Auxiliary functions for handling data and formatting for PCA plot input.

These functions extract PCA coordinates, explained variance, and loadings,
and organize them into DataFrames for use in scatter plotting.

"""

import logging

import anndata as ad
import numpy as np
import pandas as pd

# logging configuration
logging.basicConfig(level=logging.INFO)

## Helper function to validate plots inputs


def _validate_adata_and_dim_space(adata, dim_space: str) -> None:  # noqa: ANN001
    """
    Validate that data is an AnnData object and dim_space is either 'obs' or 'var'.

    Parameters
    ----------
    adata : object
        The object to check for AnnData type.
    dim_space : str
        The dimension space, must be 'obs' or 'var'.

    Raises
    ------
    TypeError
        If adata is not an AnnData object.
    ValueError
        If dim_space is not 'obs' or 'var'.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("data must be an AnnData object")

    if dim_space not in ["obs", "var"]:
        raise ValueError(f"dim_space must be either 'obs' or 'var', got {dim_space}")


def _validate_pca_plot_input(
    adata: ad.AnnData,
    pca_embeddings_layer_name: str,
    pca_var_key: str,
    dim_space: str,
) -> None:
    """
    Validates the AnnData object for PCA-related data and dimensions.

    Parameters
    ----------
    adata:
        AnnData object to be validated.
    pca_embeddings_layer_name:
        Name of the PCA layer to be checked.
    pca_var_key:
        Name of the PCA variance metadata layer to be checked, stored in `data.uns`.
    dim_space:
        The dimension space used in PCA. Can be either "obs" or "var".
    """
    _validate_adata_and_dim_space(adata, dim_space)

    # Determine which attribute to check based on dim_space
    pca_coors_attr = "obsm" if dim_space == "obs" else "varm"

    # Check if the PCA embeddings layer exists in the correct attribute
    if pca_embeddings_layer_name not in getattr(adata, pca_coors_attr):
        available_layers = list(getattr(adata, pca_coors_attr).keys())
        raise ValueError(
            f"PCA embeddings layer '{pca_embeddings_layer_name}' not found in data.{pca_coors_attr}"
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
    dim_space: str,
    pca_variance_layer_name: str,
) -> None:
    """
    Validate inputs for scree plot of the PCA dimension.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing PCA results.
    n_pcs : int
        The number of principal components requested for plotting.
    dim_space : str
        The dimension space used in PCA. Can be either "obs" or "var".
    pca_variance_layer_name : str
        The name of the PCA layer (used to construct the embedding key as `data.uns[pca_name]`).


    """
    _validate_adata_and_dim_space(adata, dim_space)

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
    adata: ad.AnnData, loadings_name: str, dim: int, dim2: int | None, nfeatures: int, dim_space: str
) -> None:
    """
    Validate inputs for accessing PCA feature loadings from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing PCA loadings data.
    loadings_name : str
        The key that stores PCA feature loadings (e.g., "PCs").
    dim: int
        The principal component index (1-based) to extract loadings for.
    dim2 : int | None
        The second principal component index (1-based) to extract loadings for, if applicable.
    nfeatures : int
        The number of top features to consider for the given component.
    dim_space : str
        The dimension space used in PCA. Can be either "obs" or "var".
    """
    _validate_adata_and_dim_space(adata, dim_space)

    # Determine which attribute to check based on dim_space
    loadings_attr = "varm" if dim_space == "obs" else "obsm"

    # Check if the loadings layer exists in the correct attribute
    if loadings_name not in getattr(adata, loadings_attr):
        available_layers = list(getattr(adata, loadings_attr).keys())
        raise ValueError(
            f"PCA feature loadings layer '{loadings_name}' not found in adata.{loadings_attr} "
            f"Found layers: {available_layers}"
        )

    # Check PC dimensions
    n_pcs = getattr(adata, loadings_attr)[loadings_name].shape[1]
    if not (1 <= dim <= n_pcs):
        raise ValueError(f"PC must be between 1 and {n_pcs} (inclusive). Got {dim=}")
    if dim2 is not None and not (1 <= dim2 <= n_pcs):
        raise ValueError(f"second PC must be between 1 and {n_pcs} (inclusive). Got pc_y={dim2}")

    # Check number of features
    n_features = getattr(adata, loadings_attr)[loadings_name].shape[0]
    if not (1 <= nfeatures <= n_features):
        raise ValueError(f"Number of features must be between 1 and {n_features} (inclusive). Got {nfeatures=}")


## Functions to prepare data frames for plotting using the scatter method


def _extract_expression_df(
    adata: ad.AnnData,
    names: list[str] | str | None,
) -> pd.DataFrame:
    """
    Extract expression data from an AnnData object as a numeric DataFrame.

    Parameters
    ----------
    adata : ad.AnnData
        Source AnnData object.
    names : list of str
        Variable names to extract.

    Returns
    -------
    pandas.DataFrame
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
    dim_space: str = "obs",
    embeddings_name: str | None = None,
    expression_columns: list[str] | None = None,
) -> ad.AnnData:
    """
    Extract PCA data required for PCA plotting from an AnnData object.

    Parameters
    ----------
    data : ad.AnnData
        AnnData object containing PCA results.
    dim_space : str
        Either "obs" or "var", indicating the PCA projection space.
    embeddings_name : str or None
        Custom embeddings name or None to use the default naming scheme.
    expression_columns : list of str or None
        List of `var_names` to include as additional numerical column(s) in
        the returned AnnData's `.obs` for coloring PCA plots by expression.
        Note that this is only applicable when `dim_space="obs"`, as there's no
        equivalent in observations when projecting in feature space (`dim_space="var"`).

    Returns
    -------
    ad.AnnData
        An AnnData object containing the PCA results.
        - `.X` stores the PCA embeddings:
            - shape (observations x components) if `dim_space="obs"`
            - shape (variables x components) if `dim_space="var"`
        - `.var` contains the PCA variance information
        - `.obs` contains the corresponding metadata, and, if specified,
          additional expression values for coloring plots.
        - PCA dimensions in `.var_names` are named as `pc_1`, `pc_2`, etc.
    """
    # Resolve PCA keys
    pca_coors_key = f"X_pca_{dim_space}" if embeddings_name is None else embeddings_name
    pca_var_key = f"variance_pca_{dim_space}" if embeddings_name is None else embeddings_name

    # Validate inputs
    _validate_pca_plot_input(adata, pca_coors_key, pca_var_key, dim_space)

    # Select PCA coordinates and metadata
    if dim_space == "obs":
        pca_coordinates = adata.obsm[pca_coors_key]
        obs_df = adata.obs

        # Add expression columns if provided
        if expression_columns is not None:
            expr_data = _extract_expression_df(
                adata=adata,
                names=expression_columns,
            )
            obs_df = obs_df.join(expr_data)

    else:  # dim_space == "var":
        pca_coordinates = adata.varm[pca_coors_key]
        obs_df = adata.var

    var_df = pd.DataFrame(adata.uns[pca_var_key])

    # Initialize PCA AnnData
    adata_pca = ad.AnnData(X=pca_coordinates)
    adata_pca.obs = obs_df.copy()
    adata_pca.var = var_df.copy()

    # Name PCA dimensions
    adata_pca.var_names = [f"pc_{i + 1}" for i in range(adata_pca.X.shape[1])]

    return adata_pca


def prepare_scree_data_to_plot(
    adata: ad.AnnData, n_pcs: int, dim_space: str, embeddings_name: str | None = None
) -> pd.DataFrame:
    """
    Prepare scree plot data from AnnData object.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing PCA results.
    n_pcs : int
        Number of principal components to include.
    dim_space : str
        The dimension space used in PCA. Can be either "obs" or "var".
    embeddings_name : str | None
        Custom embeddings name or None for default.

    Returns
    -------
    pd.DataFrame
        DataFrame with PC numbers and explained variance values.
    """
    # Generate the correct variance key name
    variance_key = f"variance_pca_{dim_space}" if embeddings_name is None else embeddings_name

    # Input checks
    _validate_scree_plot_input(adata, n_pcs, dim_space, variance_key)

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
    data: ad.AnnData | pd.DataFrame,
    dim_space: str,
    dim: int,
    nfeatures: int,
    embeddings_name: str | None = None,
) -> pd.DataFrame:
    """Prepare the gene loadings (1d) of a PC for plotting.

    Parameters
    ----------
    data : ad.AnnData
        AnnData to plot.
    dim_space : str, optional
        The dimension space used in PCA. Can be either "obs" (default) for sample projection or "var" for feature projection. By default "obs".
    dim : int
        The PC number from which to get loadings (1-indexed, i.e. the first PC is 1, not 0).
    nfeatures : int
        The number of top absolute loadings features to plot.
    embeddings_name : str | None, optional
        The custom embeddings name used in PCA. If None, uses default naming convention. By default None.

    Returns
    -------
    dataframe
        DataFrame containing the top nfeatures loadings for the specified PC dimension.

    """
    # Generate the correct loadings key name
    loadings_key = f"PCs_{dim_space}" if embeddings_name is None else embeddings_name

    # Determine which attribute to use for loadings based on dim_space
    loadings_attr = "varm" if dim_space == "obs" else "obsm"

    _validate_pca_loadings_plot_inputs(
        data=data, loadings_name=loadings_key, dim=dim, dim2=None, nfeatures=nfeatures, dim_space=dim_space
    )

    # create the dataframe for plotting
    dim_z = dim - 1  # to account from 0 indexing
    loadings_matrix = getattr(data, loadings_attr)[loadings_key]
    loadings_df = pd.DataFrame({"dim_loadings": loadings_matrix[:, dim_z]})

    # Use appropriate index for features based on dim_space
    if dim_space == "obs":
        loadings_df["feature"] = data.var.index.astype("string")
    else:  # dim_space == "var"
        loadings_df["feature"] = data.obs.index.astype("string")

    loadings_df["abs_loadings"] = loadings_df["dim_loadings"].abs()
    # Sort the DataFrame by absolute loadings and select the top features
    top_loadings_df = loadings_df.sort_values(by="abs_loadings", ascending=False).copy().head(nfeatures)
    top_loadings_df = top_loadings_df.reset_index(drop=True)
    top_loadings_df["index_int"] = range(nfeatures, 0, -1)

    return top_loadings_df


def prepare_pca_2d_loadings_data_to_plot(
    data: ad.AnnData, loadings_name: str, pc_x: int, pc_y: int, nfeatures: int, dim_space: str
) -> pd.DataFrame:
    """
    Prepare a DataFrame with PCA feature loadings for the 2D plotting.

    This function extracts the loadings of two specified principal components (PCs) from
    an AnnData object, filters features that contributed to the PCA (non-zero loadings),
    and flags the top nfeatures for each selected PC dimension.

    Parameters
    ----------
    data : anndata.AnnData
        The AnnData object containing PCA results.
    loadings_name : str
        The key where PCA loadings are stored.
    pc_x : int
        The first principal component index (1-based) to extract loadings for.
    pc_y : int
        The second principal component index (1-based) to extract loadings for.
    nfeatures : int
        Number of top features per PC to highlight based on absolute loadings.
    dim_space : str
        The dimension space used in PCA. Can be either "obs" or "var".

    Returns
    -------
    pd.DataFrame
        DataFrame containing loadings for the selected PCs, feature names, boolean columns
        indicating if a feature was used in PCA and whether it is among the top features in either dimension.
    """
    _validate_pca_loadings_plot_inputs(
        data=data, loadings_name=loadings_name, dim=pc_x, dim2=pc_y, nfeatures=nfeatures, dim_space=dim_space
    )

    dim1_z = pc_x - 1  # convert to 0-based index
    dim2_z = pc_y - 1  # convert to 0-based index

    # Determine which attribute to use based on dim_space
    loadings_attr = "varm" if dim_space == "obs" else "obsm"
    orig_loadings = getattr(data, loadings_attr)[loadings_name]

    loadings = pd.DataFrame(
        {
            "dim1_loadings": orig_loadings[:, dim1_z],
            "dim2_loadings": orig_loadings[:, dim2_z],
        }
    )

    # Add feature names based on dim_space
    if dim_space == "obs":
        loadings["feature"] = data.var_names
    else:  # dim_space == "var"
        loadings["feature"] = data.obs_names

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
