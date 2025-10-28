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

    If axis is 0, assume metadata.index <-> data.index and add metadata as '.obs' of the AnnData object.
    If axis is 1, assume metadata.index <-> data.columns and add metadata as '.var' of the AnnData object.

    Parameters
    ----------
        adata : ad.AnnData
                Anndata object to add metadata to.
        incoming metadata : pd.DataFrame
                Metadata dataframe to add. The matching entity is always the INDEX, depending on axis it is
                matched against obs (axis = 0) or var (axis = 1).
        axis : int
                Axis to add metadata to. 0 for obs and 1 for var.
        keep_data_shape : bool = False
                If True, the incoming data is left-joined to the existing data, which may result in nan-padded
                rows in the incoming data. If False, incoming data is added via inner join, which may change the
                shape of the adata object.
        keep_existing_metadata : bool = False
                If True, incoming metadata is added to the existing metadata. If False, incoming metadata replaces
                existing metadata. If columns between existing and incoming metadata are synonymous, the corresponding
                incoming metadata columns are ignored.
        verbose : bool = False
                If True, print additional information about the operation.


    """
    # Basic checks
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1.")

    if not isinstance(incoming_metadata, pd.DataFrame) or incoming_metadata.index.nlevels > 1:
        raise TypeError("metadata must be a pd.DataFrame with single-level index.")

    if any(incoming_metadata.index.duplicated()):
        raise ValueError("Duplicated metadata indices are not supported.")

    _raise_nonoverlapping_indices(adata.to_df(), incoming_metadata, axis)

    # set join type
    join = "left" if keep_data_shape else "inner"

    ### Handle alignment of incoming and existing metadata
    if keep_existing_metadata:
        existing_metadata = adata.obs if axis == 0 else adata.var

        # if existing metadata should be kept and new metadata contains synonymous fields to existing metadata, drop incoming fields
        incoming_metadata = _handle_overlapping_columns(incoming_metadata, existing_metadata, verbose=verbose)

        # join new to existing metadata; same join logic as for data
        if verbose:
            logging.info(f"pp.add_metadata(): Join incoming to existing metadata via {join} join on axis  {axis}.")

        incoming_metadata = existing_metadata.join(incoming_metadata, how=join)

    ### Emulate join on AnnData level without copying
    # 1. Reindex the AnnData object for inner join
    if join == "inner":
        existing_fields = adata.obs.index if axis == 0 else adata.var.index
        shared_fields = existing_fields.intersection(incoming_metadata.index)
        adata = adata[shared_fields, :] if axis == 0 else adata[:, shared_fields]

    # 2. Align the new metadata to obs or var of the AnnData object
    if axis == 0:
        incoming_metadata = incoming_metadata.reindex(adata.obs.index)
    elif axis == 1:
        incoming_metadata = incoming_metadata.reindex(adata.var.index)

    # 3. use the [] method to subset the adata object inplace based on the obs and incoming indices
    if axis == 0:
        bool_mask = adata.obs.index.isin(incoming_metadata.index)
        adata = adata[bool_mask, :]
    elif axis == 1:
        bool_mask = adata.var.index.isin(incoming_metadata.index)
        adata = adata[:, bool_mask]

    # 4. reindex the incoming metadata to match the adata object's obs or var index
    if axis == 0:
        incoming_metadata = incoming_metadata.reindex(adata.obs.index)
    elif axis == 1:
        incoming_metadata = incoming_metadata.reindex(adata.var.index)

    # 5. assign the new metadata to the adata object's obs or var attribute
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


def data_column_to_array(
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
            logging.info(f"Column '{column}' found in: data.var_names. Using that")
            return data.X[:, col_idx].flatten()

        if column in data.obs.columns:
            logging.info(f"Column '{column}' found in: data.obs.columns. Using that")
            return data.obs[column].to_numpy()

        if column in data.var.columns:
            logging.info(f"Column '{column}' found in: data.var.columns. Using that")
            return data.var[column].to_numpy()

        raise ValueError(
            f"Column {column} not found in AnnData object (checked var_names and obs.columns and var.columns)."
        )

    raise TypeError(f"Expected pd.DataFrame or ad.AnnData, got {type(data)}")


def scale_and_center(  # explicitly tested via test_pp_scale_and_center()
    adata: ad.AnnData, scaler: str = "standard", layer: str | None = None, *, copy: bool = False
) -> None | ad.AnnData:
    """Scale and center data.

    Either use standard or robust scaling. 'robust' scaling relies
    on interquartile range and is more resistant to outliers. Scaling
    operates on columns only for now.

    Parameters
    ----------
    adata
        Anndata object with data to scale.
    scaler
        Sklearn scaler to use. Available scalers are 'standard' and 'robust'.
    layer
        Name of the layer to scale. If None (default), the data matrix X is used.
    copy
        Whether to return a modified copy (True) of the anndata object. If False (default)
        modifies the object inplace

    Returns
    -------
    None | anndata.AnnData
        If `copy=False` modifies the anndata object at layer inplace and returns None. If `copy=True`,
        returns a modified copy.
    """
    adata = adata.copy() if copy else adata
    logging.info(f"pp.scale_and_center(): Scaling data with {scaler} scaler.")

    if scaler == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    elif scaler == "robust":
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    else:
        raise NotImplementedError(f"Scaler {scaler} not implemented.")

    input_data = adata.X if layer is None else adata.layers[layer]
    result = scaler.fit_transform(input_data)
    if layer is None:
        adata.X = result
    else:
        adata.layers[layer] = result

    return adata if copy else None


# TODO: Abstract class for validation of AnnData objects?
def _validate_adata_for_completeness_filter(
    adata: ad.AnnData,
) -> None:
    """Validate AnnData object for data completeness filtering"""
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an AnnData object.")

    if adata.shape[1] == 0:
        raise ValueError("adata has no features (columns).")

    if not is_numeric_dtype(adata.X):
        raise ValueError("adata.X must be numeric.")

    if any(adata.obs.index.duplicated()):
        raise ValueError("pp.filter_data_completeness(): Duplicated indices in obs")


def filter_data_completeness(
    adata: ad.AnnData,
    max_missing: float,
    group_column: str | None = None,
    groups: list[str] | None = None,
) -> ad.AnnData:
    """Filter features based on missing values

    Filters AnnData features (columns) based on the fraction of missing values.
    If group_column and groups are provided, only missingness of certain metadata
    levels is considered. This is especially useful for imbalanced classes, where
    filtering by global missingness may leave too many missing values in the smaller
    class.

    (In case rows should be filtered, it is recommended to transpose the adata
    object prior to calling this function and reverting the transpose afterwards.)

    Parameters
    ----------
    max_missing : float
        Maximum fraction of missing values allowed. Compared with the fraction of missing values
        in a "greater than" fashion, i.e. if max_missing is 0.6 and the fraction of missing values
        is 0.6, the sample or feature is kept. Greater than comparison is used here since the
        missing fraction may be 0.0, in which case the sample or feature should be kept.
    group_column : str, optional
        Column in obs to determine groups for filtering.
    groups : list[str], optional
        List of levels of the group_column to consider in filtering. E.g. if the column has the levels
        ['A', 'B', 'C'], and groups = ['A', 'B'], only missingness of features in these
        groups is considered. If None, all groups are considered.

    """
    if max_missing < 0 or max_missing > 1:
        raise ValueError("Threshold must be between 0 and 1.")

    _validate_adata_for_completeness_filter(adata)

    # Resolve group indices
    if group_column:
        if group_column not in adata.obs.columns:
            raise ValueError(f"Group column '{group_column}' not found in obs, available: {adata.obs.columns}.")

        available_groups = set(adata.obs[group_column].unique())
        selected_groups = set(groups) if groups else available_groups

        if not selected_groups.issubset(available_groups):
            raise ValueError(f"Some groups in {groups} not found in '{group_column}'.")

        group_indices = {group: adata.obs.index[adata.obs[group_column] == group] for group in selected_groups}
    else:
        group_indices = {"all": adata.obs.index}

    # Calculate missingness for each group
    drop = np.array([False] * adata.shape[1])
    for indices in group_indices.values():
        missing_fraction = np.isnan(adata[indices, :].X).mean(axis=0)
        drop |= missing_fraction > max_missing

    # Drop columns with too many missing values from data
    if drop.any():
        adata = adata[:, ~drop].copy()
        n_dropped = drop.sum()
        logging.info(
            f"pp.filter_data_completeness(): Dropped {n_dropped} / {drop.size} features with >{max_missing:.2f} missing in any group."
        )

    return adata
