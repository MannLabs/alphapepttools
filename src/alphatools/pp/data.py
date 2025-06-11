# Create and manipulate Anndata objects

import logging
import numbers
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


def _filter_by_dict(
    data: pd.DataFrame,
    filter_dict: dict,
    logic: str = "and",
) -> pd.Series:
    """Core filtering function to operate on metadata

    Versatile filtering for pd.DataFrames, where different filtering logic
    can be applied: The filter_dict contains keys, which are column names,
    and values, which can be either strings, lists or tuples (see below).
    The 'logic' parameter determines whether multiple filters operate on
    an 'and' or 'or' basis. Returns indices of samples that match the filter
    conditions.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to filter. Columns must match with filter_dict keys.
    filter_dict : dict
        Dictionary with column names as keys and filter values as values.
        Values can be either string, list or tuple. For strings, exact matches
        are performed. For lists, matches are performed on any element in the
        list. Tuples specify value ranges and must consist of numeric values,
        where 'None' is interpreted as an open end. Ranges are inclusive on
        the lower end and exclusive on the upper end to prevent double counting
        with adjacent filters.
    logic : str, optional
        Filtering logic to apply in case of multiple filters. Default to 'and'.
        Can be 'and' or 'or'.

    Returns
    -------
    filter_mask : pd.Series
        Boolean mask to filter the dataframe.

    """
    # data must have unique indices
    if any(data.index.duplicated()):
        raise ValueError("pp.filter_by_dict(): Duplicated indices in data, reassigning index.")

    if not filter_dict:
        return pd.Series(True, index=data.index)  # noqa: FBT003

    _verify_filter_dict(filter_dict, data)

    filter_masks = []
    for k, v in filter_dict.items():
        feature = data[k] if k != "index" else data.index
        if v is None:
            current_mask = pd.Series(True, index=data.index)  # noqa: FBT003
        elif isinstance(v, str | numbers.Number):
            current_mask = feature == v
        elif isinstance(v, list):
            current_mask = feature.isin(v)
        elif isinstance(v, tuple):
            current_mask = _tuple_based_filter(feature, v)

        filter_masks.append(current_mask)

    # if 'and', all masks must be True to keep a row
    # if 'or', at least one mask must be True to keep a row
    if logic == "and":
        merged_filter_mask = np.all(filter_masks, axis=0)
    elif logic == "or":
        merged_filter_mask = np.any(filter_masks, axis=0)

    return merged_filter_mask


def _tuple_based_filter(
    feature: pd.Series,
    input_tuple: tuple,
) -> pd.Series:
    """Tuple-based filtering of numeric features"""
    errors = []
    if not is_numeric_dtype(feature):
        errors.append("Tuple-based filtering only works on numeric features.")
    if len(input_tuple) != 2:  # noqa: PLR2004
        errors.append("Tuple-based filtering requires a tuple of length 2.")
    if not all(isinstance(x, numbers.Number) or x is None for x in input_tuple):
        errors.append("Tuple-based filtering requires numeric values or None.")

    if errors:
        raise ValueError("Errors found in tuple-based filtering:\n" + "\n".join(errors))

    lower, upper = input_tuple
    if lower is not None and upper is not None:
        current_mask = (feature >= lower) & (feature < upper)
    elif lower is not None:
        current_mask = feature >= lower
    elif upper is not None:
        current_mask = feature < upper
    else:
        current_mask = pd.Series(True, index=feature.index)  # noqa: FBT003

    return current_mask


def _verify_filter_dict(
    filter_dict: dict,
    data: pd.DataFrame,
) -> None:
    errors = []
    for k, v in filter_dict.items():
        if not isinstance(k, str):
            errors.append(f"Filter keys must be string, not {type(k).__name__}.")
        if k not in data.columns and k != "index":
            errors.append(f"Filter key '{k}' is not 'index' and also not found in data columns.")
        if not isinstance(v, str | numbers.Number | list | tuple):
            errors.append(f"Filter values must be of type str, number, list or tuple, not {type(v)}.")

    if errors:
        raise ValueError("Errors found in filter_dict:\n" + "\n".join(errors))


def filter_by_metadata(
    adata: ad.AnnData,
    filter_dict: dict,
    axis: int,
    logic: str = "and",
    action: str = "keep",
) -> ad.AnnData:
    """Filter based on metadata

    Filter or drop rows/columns from an adata object based on filter conditions
    specified in a filter_dict. The filter_dict contains keys, which are column
    names, and values, which can be either strings, lists or tuples. The 'logic'
    parameter determines whether multiple filters operate on an 'and' or 'or'
    basis.

    Parameters
    ----------
    adata : ad.AnnData
        Anndata object to filter.
    filter_dict : dict
        Dictionary with column names as keys and filter values as values.
        Values can be either string, list or tuple. For strings, exact matches
        are performed. For lists, matches are performed on any element in the
        list. Tuples specify value ranges and must consist of numeric values,
        where 'None' is interpreted as an open end. Ranges are inclusive on
        the lower end and exclusive on the upper end to prevent double counting
        with adjacent filters.
    axis : int
        Axis to filter on. 0 for obs and 1 for var.
    logic : str, optional
        Filtering logic to apply in case of multiple filters. Default to 'and'.
        Can be 'and' or 'or'.
    action : str, optional
        If "keep", extract rows/columns that match the filter conditions.
        If "drop", extract rows/columns outside the filter conditions.
        Default to "keep".

    Returns
    -------
    adata : ad.AnnData
        Filtered anndata object.

    """
    metadata_to_filter = adata.obs if axis == 0 else adata.var
    filter_mask = _filter_by_dict(metadata_to_filter, filter_dict, logic)

    if action == "drop":
        filter_mask = ~filter_mask

    if axis == 0:
        adata = adata[filter_mask, :]
    elif axis == 1:
        adata = adata[:, filter_mask]
    else:
        raise ValueError("Invalid 'axis' parameter, must be 0 or 1.")

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
    raise TypeError("Data must be a pd.DataFrame or ad.AnnData.")


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


# def filter_data_completeness(
#     adata: ad.AnnData,
#     max_missing: float,
#     group_column: str | None = None,
#     groups: list[str] | None = None,
# ) -> ad.AnnData:
#     """Filter data based on missing values

#     Filter either features based on the fraction of missing values.
#     If group_column and groups are provided, only missingness of certain metadata
#     levels is considered. This is especially useful for imbalanced classes, where
#     filtering by global missingness may leave too many missing values in the smaller
#     class.

#     Parameters
#     ----------
#     max_missing : float
#         Maximum fraction of missing values allowed. Compared with the fraction of missing values
#         in a "greater than" fashion, i.e. if max_missing is 0.6 and the fraction of missing values
#         is 0.6, the sample or feature is kept. Greater than comparison is used here since the
#         missing fraction may be 0.0, in which case the sample or feature should be kept.
#     group_column : str, optional
#         Column in obs to determine groups for filtering.
#     groups : list[str], optional
#         List of levels of the group_column to consider in filtering. E.g. if the column has the levels
#         ['A', 'B', 'C'], and groups = ['A', 'B'], only missingness of features in these
#         groups is considered. If None, all groups are considered.

#     """
#     if max_missing < 0 or max_missing > 1:
#         raise ValueError("Threshold must be between 0 and 1.")

#     if not is_numeric_dtype(adata.X):
#         raise ValueError("Data must be numeric.")

#     # Resolve group indices
#     group_indices = {}
#     if group_column is not None:
#         if group_column not in adata.obs.columns:
#             raise ValueError(f"Group column '{group_column}' not found in obs.")
#         group_column_values = adata.obs[group_column].unique()
#         if groups is not None:
#             if not set(groups).issubset(set(group_column_values)):
#                 raise ValueError(f"Groups {groups} not found in group column '{group_column}'.")
#             for group in np.unique(groups):
#                 group_indices[group] = adata.obs.index[adata.obs[group_column] == group]
#         else:
#             for group in group_column_values:
#                 group_indices[group] = adata.obs.index[adata.obs[group_column] == group]
#     else:
#         group_indices["all"] = adata.obs.index

#     # Calculate missingness for each group
#     drop = np.array([False] * adata.shape[1])
#     for group, indices in group_indices.items():
#         missing_fraction = adata[indices, :].X.isnull().mean(axis=0)
#         drop |= missing_fraction > max_missing

#     # Drop columns with too many missing values from data
#     if drop.any():
#         adata = adata[:, ~drop].copy()
#         logging.info(
#             f"pp.filter_data_completeness(): Dropped {drop.size} features with >{max_missing:.2f} missing values."
#         )

#     return adata
