# Create and manipulate Anndata objects

import logging
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import RobustScaler, StandardScaler

# logging configuration
logging.basicConfig(level=logging.INFO)


### PLACEHOLDER FOR ALPHABASE DIANN-READER WRAPPERS ###
def load_diann_pg_matrix(
    data_path: str,
) -> ad.AnnData:
    """Placeholder for development; load diann sample data into a pandas dataframe"""
    X = pd.read_pickle(data_path)

    # to be replaced by AlphaBase PSM reader
    return _to_anndata(X)


def _to_anndata(
    data: np.ndarray | pd.DataFrame,
) -> ad.AnnData:
    """Create an AnnData object from a data array and optional sample & feature metadata

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
            The data array, where rows correspond to samples and columns correspond to features.
            If data is a pd.DataFrame, row and column indices are used for inner join with obs and var.
            If data is a np.ndarray, obs and var are not assigned.

    Returns
    -------
    adata : ad.AnnData
            Anndata object with data, obs, and var.

    """
    # If data is a dataframe, convert row and col indices to obs and var
    if isinstance(data, pd.DataFrame):
        adata = ad.AnnData(data.values)
        adata.obs = data.index.to_frame(name="obs")
        adata.var = data.columns.to_frame(name="var")
    elif isinstance(data, np.ndarray):
        adata = ad.AnnData(data)

    return adata


def add_metadata(  # noqa: C901, PLR0912
    adata: ad.AnnData,
    incoming_metadata: pd.DataFrame,
    axis: int,
    *,
    keep_data_shape: bool = False,
    keep_existing_metadata: bool = False,
    verbose: bool = False,
) -> ad.AnnData:
    """Add metadata to an AnnData object while checking for matching indices or shape

        If axis is 0, assume metadata.index <-> data.index and add metadata as '.obs' attribute of the AnnData object.
    If axis is 1, assume metadata.index <-> data.columns and add metadata as '.var' attribute of the AnnData object.

    Parameters
    ----------
        adata : ad.AnnData
                Anndata object to add metadata to.

        metadata : pd.DataFrame
                Metadata dataframe to add. The matching entity is always the INDEX, depending on axis it is
                matched against obs (axis = 0) or var (axis = 1). Correspondingly, the data-aligned dimension
                of both adata.obs and adata.var is always the index.

    """
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1.")

    if not isinstance(incoming_metadata, pd.DataFrame) or incoming_metadata.index.nlevels > 1:
        raise TypeError("metadata must be a pd.DataFrame with single-level index.")

    if any(incoming_metadata.index.duplicated()):
        raise ValueError("Duplicated metadata indices are not supported.")

    _raise_nonoverlapping_indices(adata.to_df(), incoming_metadata, axis)

    # set join type
    join = "left" if keep_data_shape else "inner"

    # if existing metadata should be kept and new metadata contains synonymous fields to existing metadata, drop incoming fields
    if keep_existing_metadata:
        if axis == 0:
            inplace_metadata = adata.obs
        elif axis == 1:
            inplace_metadata = adata.var

        # handle overlapping metadata columns
        incoming_metadata = _handle_overlapping_columns(incoming_metadata, inplace_metadata, verbose=verbose)

        # join new to existing metadata; same join logic as for data
        if verbose:
            logging.info(f"pp.add_metadata(): Join incoming to existing metadata via {join} join on axis  {axis}.")
        incoming_metadata = inplace_metadata.join(incoming_metadata, how=join)

    # TODO: streamline logic below
    # 1. align the new metadata to obs or var under the proper join
    if axis == 0:
        incoming_metadata = incoming_metadata.reindex(adata.obs.index)
    elif axis == 1:
        incoming_metadata = incoming_metadata.reindex(adata.var.index)

    # 2. use the [] method to subset the adata object inplace based on the obs and incoming indices
    if axis == 0:
        bool_mask = adata.obs.index.isin(incoming_metadata.index)
        adata = adata[bool_mask, :]
    elif axis == 1:
        bool_mask = adata.var.index.isin(incoming_metadata.index)
        adata = adata[:, bool_mask]

    # 3. reindex the incoming metadata to match the adata object's obs or var index
    if axis == 0:
        incoming_metadata = incoming_metadata.reindex(adata.obs.index)
    elif axis == 1:
        incoming_metadata = incoming_metadata.reindex(adata.var.index)

    # 4. assign the new metadata to the adata object's obs or var attribute
    if axis == 0:
        if not adata.obs.index.equals(incoming_metadata.index):
            raise ValueError("Index mismatch between data and metadata.")
        adata.obs = incoming_metadata
    elif axis == 1:
        if not adata.var.index.equals(incoming_metadata.index):
            raise ValueError("Index mismatch between data and metadata.")
        adata.var = incoming_metadata

    return adata


def _raise_nonoverlapping_indices(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    axis: int,
) -> None:
    """Check if any fields overlap between two dataframes on respective axes"""
    if axis == 0:
        shared_idx_len = len(data.index.intersection(metadata.index))
    elif axis == 1:
        shared_idx_len = len(data.columns.intersection(metadata.index))

    if shared_idx_len == 0:
        raise ValueError(f"No matching fields found between data and metadata (axis = {axis}).")


# TODO: Add test for this function or refactor to handle indices with suffixes
def _handle_overlapping_columns(
    metadata: pd.DataFrame,
    _inplace_metadata: pd.DataFrame,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop overlapping fields from incoming metadata to avoid name collisions"""
    overlapping_fields = metadata.columns.intersection(_inplace_metadata.columns)
    if not overlapping_fields.size:
        return metadata
    if verbose:
        warnings.warn(
            f"pp.add_metadata(): Synonymous fields, dropping {overlapping_fields.to_list()} from incoming metadata."
        )
    return metadata.drop(
        overlapping_fields,
        axis=1,
        errors="ignore",
        inplace=False,
    )


def _adata_column_to_array(
    data: pd.DataFrame | ad.AnnData,
    column: str,
) -> np.ndarray:
    """Get a column from a DataFrame or an AnnData object

    Parameters
    ----------
    data : pd.DataFrame | ad.AnnData
        Data to extract the column from.
    column : str
        Column name to extract. If data is of type ad.AnnData, var_names is considered
        first for the column names. If the column is not found in var_names, the columns
        of data.obs are considered. If the column is not found in either, a ValueError
        is raised.

    Returns
    -------
    np.ndarray
        The column data as a numpy array

    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column {column} not found in DataFrame.")
        return data[column].to_numpy()
    if isinstance(data, ad.AnnData):
        # prioritize var_names, i.e. numeric data from X
        if column in data.var_names:
            col_idx = data.var_names.get_loc(column)
            return data.X[:, col_idx].flatten()

        # if the column is not found in var_names, check the columns of obs (metadata)
        if column in data.obs.columns:
            return data.obs[column].to_numpy()

        raise ValueError(f"Column {column} not found in AnnData object (checked var_names or obs.columns).")
    raise TypeError("Data must be a pandas DataFrame or an AnnData object.")


def scale_and_center(  # explicitly tested via test_pp_scale_and_center()
    adata: ad.AnnData,
    scaler: str = "standard",
    from_layer: str | None = None,
    to_layer: str | None = None,
) -> None:
    """Scale and center data.

    Either use standard or robust scaling. 'robust' scaling relies
    on interquartile range and is more resistant to outliers. Scaling
    operates on columns only for now.

    Parameters
    ----------
    adata : ad.AnnData
        Anndata object with data to scale.
    scaler : str
        Sklearn scaler to use. Available scalers are 'standard' and 'robust'.
    from_layer : str, optional
        Name of the layer to scale. If None, the data matrix X is used.
    to_layer : str, optional
        Name of the layer to scale. If None, the data matrix X is modified

    Returns
    -------
    None
    """
    mod_status = "inplace" if to_layer is None else f"to layer '{to_layer}'"

    logging.info(f"pp.scale_and_center(): Scaling data with {scaler} scaler {mod_status}.")

    if scaler == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    elif scaler == "robust":
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    else:
        raise NotImplementedError(f"Scaler {scaler} not implemented.")

    input_data = adata.X if from_layer is None else adata.layers[from_layer]
    result = scaler.fit_transform(input_data)
    if to_layer is None:
        adata.X = result
    else:
        adata.layers[to_layer] = result


def filter_data_completeness(
    adata: ad.AnnData,
    max_missing: float,
    group_column: str | None = None,
    groups: list[str] | None = None,
    axis: int = 0,
) -> ad.AnnData:
    """Filter data based on missing values

    Filter either samples or features based on the fraction of missing values.
    ### NOT IMPLEMENTED YET: Group-based filtering ###
    If group_column and groups are provided, only missingness of certain metadata
    levels is considered. This is especially useful for imbalanced classes, where
    filtering by global missingness may leave too many missing values in the smaller
    class.

    Parameters
    ----------
    max_missing : float
        Maximum fraction of missing values allowed. Compared with the fraction of missing values
        in a "greater than" fashion, i.e. if max_missing is 0.6 and the fraction of missing values
        is 0.6, the sample or feature is kept. Greater than comparison is used here since the
        missing fraction may be 0.0, in which case the sample or feature should be kept.
    group_column : str, optional
        Column in obs or var to determine groups for filtering.
    groups : list[str], optional
        List of groups to consider in filtering.
    axis : int, optional
        Whether to check completeness of samples (0) or features (1).

    """
    if max_missing < 0 or max_missing > 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if group_column:
        raise NotImplementedError("Group-based filtering not implemented yet.")
    if groups:
        raise NotImplementedError("Group-based filtering not implemented yet.")

    if not is_numeric_dtype(adata.X):
        raise ValueError("Data must be numeric.")

    if axis == 0:  # check completeness of samples
        missing_fraction = np.isnan(adata.X).mean(axis=1)
        missing_above_cutoff = missing_fraction > max_missing
        adata = adata[~missing_above_cutoff, :]
    elif axis == 1:  # check completeness of features
        missing_fraction = np.isnan(adata.X).mean(axis=0)
        missing_above_cutoff = missing_fraction > max_missing
        adata = adata[:, ~missing_above_cutoff]

    return adata
