# TODO: move this submodule to tl module
"""
Auxiliary functions for handling data and formatting for PCA plot input.

These functions extract PCA coordinates, explained variance, and loadings,
and organize them into DataFrames for use in scatter plotting.

"""

import logging

import anndata as ad
import numpy as np
import pandas as pd

from alphatools.pp.data import data_column_to_array

# logging configuration
logging.basicConfig(level=logging.INFO)

## Helper function to validate plots inputs


def _validate_adata_and_dim_space(data, dim_space: str) -> None:  # noqa: ANN001
    """
    Validate that data is an AnnData object and dim_space is either 'obs' or 'var'.

    Parameters
    ----------
    data : object
        The object to check for AnnData type.
    dim_space : str
        The dimension space, must be 'obs' or 'var'.

    Raises
    ------
    TypeError
        If data is not an AnnData object.
    ValueError
        If dim_space is not 'obs' or 'var'.
    """
    if not isinstance(data, ad.AnnData):
        raise TypeError("data must be an AnnData object")

    if dim_space not in ["obs", "var"]:
        raise ValueError(f"dim_space must be either 'obs' or 'var', got {dim_space}")


def _validate_pca_plot_input(
    data: ad.AnnData,
    pca_embeddings_layer_name: str,
    pc_x: int,
    pc_y: int,
    dim_space: str,
) -> None:
    """
    Validates the AnnData object for PCA-related data and dimensions.

    Parameters
    ----------
    data:
        AnnData object to be validated.
    pca_embeddings_layer_name:
        Name of the PCA layer to be checked.
    pc_x:
        First PCA dimension to be validated (1-indexed, i.e. the first PC is 1, not 0).
    pc_y:
        Second PCA dimension to be validated (1-indexed, i.e. the first PC is 1, not 0).
    dim_space:
        The dimension space used in PCA. Can be either "obs" or "var".
    """
    _validate_adata_and_dim_space(data, dim_space)

    # Determine which attribute to check based on dim_space
    pca_coors_attr = "obsm" if dim_space == "obs" else "varm"

    # Check if the PCA embeddings layer exists in the correct attribute
    if pca_embeddings_layer_name not in getattr(data, pca_coors_attr):
        available_layers = list(getattr(data, pca_coors_attr).keys())
        raise ValueError(
            f"PCA embeddings layer '{pca_embeddings_layer_name}' not found in data.{pca_coors_attr}"
            f"Found layers: {available_layers}"
        )

    # Check PC dimensions
    n_pcs = getattr(data, pca_coors_attr)[pca_embeddings_layer_name].shape[1]
    if not (1 <= pc_x <= n_pcs) or not (1 <= pc_y <= n_pcs):
        raise ValueError(f"pc_x and pc_y must be between 1 and {n_pcs} (inclusive). Got {pc_x=}, {pc_y=}")


def _validate_scree_plot_input(
    data: ad.AnnData,
    n_pcs: int,
    dim_space: str,
    pca_variance_layer_name: str,
) -> None:
    """
    Validate inputs for scree plot of the PCA dimension.

    Parameters
    ----------
    data : anndata.AnnData
        The AnnData object containing PCA results.
    n_pcs : int
        The number of principal components requested for plotting.
    dim_space : str
        The dimension space used in PCA. Can be either "obs" or "var".
    pca_variance_layer_name : str
        The name of the PCA layer (used to construct the embedding key as `data.uns[pca_name]`).


    """
    _validate_adata_and_dim_space(data, dim_space)

    if pca_variance_layer_name not in data.uns:
        raise ValueError(
            f"PCA metadata layer '{pca_variance_layer_name}' not found in AnnData object. "
            f"Found layers: {list(data.uns.keys())}"
        )

    n_pcs_avail = len(data.uns[pca_variance_layer_name]["variance_ratio"])
    if n_pcs > n_pcs_avail:
        logging.warning(
            f"Requested {n_pcs} PCs, but only {n_pcs_avail} PCs are available. Plotting only the available PCs"
        )


def _validate_pca_loadings_plot_inputs(
    data: ad.AnnData, loadings_name: str, dim: int, dim2: int | None, nfeatures: int, dim_space: str
) -> None:
    """
    Validate inputs for accessing PCA feature loadings from an AnnData object.

    Parameters
    ----------
    data : anndata.AnnData
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
    _validate_adata_and_dim_space(data, dim_space)

    # Determine which attribute to check based on dim_space
    loadings_attr = "varm" if dim_space == "obs" else "obsm"

    # Check if the loadings layer exists in the correct attribute
    if loadings_name not in getattr(data, loadings_attr):
        available_layers = list(getattr(data, loadings_attr).keys())
        raise ValueError(
            f"PCA feature loadings layer '{loadings_name}' not found in data.{loadings_attr} "
            f"Found layers: {available_layers}"
        )

    # Check PC dimensions
    n_pcs = getattr(data, loadings_attr)[loadings_name].shape[1]
    if not (1 <= dim <= n_pcs):
        raise ValueError(f"PC must be between 1 and {n_pcs} (inclusive). Got {dim=}")
    if dim2 is not None and not (1 <= dim2 <= n_pcs):
        raise ValueError(f"second PC must be between 1 and {n_pcs} (inclusive). Got pc_y={dim2}")

    # Check number of features
    n_features = getattr(data, loadings_attr)[loadings_name].shape[0]
    if not (1 <= nfeatures <= n_features):
        raise ValueError(f"Number of features must be between 1 and {n_features} (inclusive). Got {nfeatures=}")


## Functions to prepare data frames for plotting using the scatter method


def prepare_pca_data_to_plot(
    data: ad.AnnData,
    pc_x: int = 1,
    pc_y: int = 2,
    dim_space: str = "obs",
    embbedings_name: str | None = None,
    color_map_column: str | None = None,
    label_column: str | None = None,
    *,
    label: bool = False,
) -> pd.DataFrame:
    """
    Fetched PCA data required from PCA plotting from AnnData object (as returned by `pca` function).

    Parameters
    ----------
    data : ad.AnnData
        AnnData object containing PCA results.
    pc_x : int
        First principal component (1-indexed).
    pc_y : int
        Second principal component (1-indexed).
    dim_space : str
        Either "obs" or "var" for observation or variable embeddings space.
    embbedings_name : str | None
        Custom embeddings name or None for default.
    color_map_column : str | None
        Column for color mapping.
    label_column : str | None
        Column for labeling points.
    label : bool
        Whether labels are requested.

    Returns
    -------
    pd.DataFrame
        DataFrame with PCA coordinates,color data and labels if requested.
    """
    # Generate the correct key names based on dim_space and embbedings_name
    pca_coors_key = f"X_pca_{dim_space}" if embbedings_name is None else embbedings_name

    # Input checks
    _validate_pca_plot_input(data, pca_coors_key, pc_x, pc_y, dim_space)

    # Create the dataframe for plotting
    dim1_z = pc_x - 1  # to account for 0 indexing
    dim2_z = pc_y - 1  # to account for 0 indexing

    # Get PCA coordinates from the correct attribute
    pca_coordinates = data.obsm[pca_coors_key] if dim_space == "obs" else data.varm[pca_coors_key]

    pca_coor_df = pd.DataFrame(pca_coordinates[:, [dim1_z, dim2_z]], columns=["dim1", "dim2"])

    # Add color column if specified
    if color_map_column is not None:
        color_values = data_column_to_array(data, color_map_column)
        pca_coor_df[color_map_column] = color_values

    # Prepare labels if requested
    labels = None
    if label:
        if dim_space == "obs":
            labels = data.obs.index if label_column is None else data_column_to_array(data, label_column)
        else:  # dim_space == "var"
            labels = data.var.index if label_column is None else data_column_to_array(data, label_column)
        pca_coor_df["labels"] = labels

    return pca_coor_df


def prepare_scree_data_to_plot(
    adata: ad.AnnData, n_pcs: int, dim_space: str, embbedings_name: str | None = None
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
    embedings_name : str | None
        Custom embeddings name or None for default.

    Returns
    -------
    pd.DataFrame
        DataFrame with PC numbers and explained variance values.
    """
    # Generate the correct variance key name
    variance_key = f"variance_pca_{dim_space}" if embbedings_name is None else embbedings_name

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
    embbedings_name: str | None = None,
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
    embbedings_name : str | None, optional
        The custom embeddings name used in PCA. If None, uses default naming convention. By default None.

    Returns
    -------
    dataframe
        DataFrame containing the top nfeatures loadings for the specified PC dimension.

    """
    # Generate the correct loadings key name
    loadings_key = f"PCs_{dim_space}" if embbedings_name is None else embbedings_name

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
