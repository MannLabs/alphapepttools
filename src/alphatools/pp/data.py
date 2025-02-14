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
        adata = ad.AnnData(data)
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

    _raise_nonoverlapping_indices(_get_df_from_adata(adata), incoming_metadata, axis)

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


def filter_by_dict(
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
        logging.warning("pp.filter_by_dict(): Duplicated indices in data, reassigning index.")
        data = data.reset_index(drop=True)

    _verify_filter_dict(filter_dict, data)

    filter_masks = []
    for k, v in filter_dict.items():
        feature = data[k] if k != "index" else data.index
        if isinstance(v, None):
            current_mask = pd.Series(True, index=data.index)  # noqa: FBT003
        elif isinstance(v, str):
            current_mask = feature == v
        elif isinstance(v, list):
            current_mask = feature.isin(v)
        elif isinstance(v, tuple):
            current_mask = _tuple_based_filter(feature, v)
        filter_masks.append(current_mask)

    # if 'and', all masks must be True to keep a row
    # if 'or', at least one mask must be True to keep a row
    if logic == "and":
        data_mask = np.all(filter_masks, axis=0)
    elif logic == "or":
        data_mask = np.any(filter_masks, axis=0)

    return data_mask


def _tuple_based_filter(
    feature: pd.Series,
    input_tuple: tuple,
) -> pd.Series:
    """Tuple-based filtering of numeric features"""
    if not is_numeric_dtype(feature):
        raise ValueError("Tuple-based filtering only works on numeric features.")
    if not len(input_tuple) == 2:  # noqa: PLR2004
        raise ValueError("Tuple-based filtering requires a tuple of length 2.")
    if not all(isinstance(x, numbers.Number) or x is None for x in input_tuple):
        raise ValueError("Tuple-based filtering requires numeric values or None.")

    lower, upper = input_tuple
    if lower is not None and upper is not None:
        current_mask = (feature >= lower) & (feature < upper)
    elif lower is not None:
        current_mask = feature >= lower
    elif upper is not None:
        current_mask = feature < upper

    return current_mask


def _verify_filter_dict(
    filter_dict: dict,
    data: pd.DataFrame,
) -> None:
    for k, v in filter_dict.items():
        if not isinstance(k, str):
            raise TypeError("Filter keys must be strings.")
        if k not in data.columns and k != "index":
            raise ValueError(f"Filter key '{k}' is not 'index' and also not found in data columns.")
        if not isinstance(v, str | list | tuple):
            raise TypeError(f"Filter values must be of type str, list or tuple, not {type(v)}.")


def filter_by_metadata(
    data: ad.AnnData,
    filter_dict: dict,
    axis: int,
) -> ad.AnnData:
    """Filter based on metadata"""


def drop_by_metadata(
    data: ad.AnnData,
    filter_dict: dict,
    axis: int,
) -> ad.AnnData:
    """Drop based on metadata"""


def _raise_nonoverlapping_indices(
    data: pd.Index,
    metadata: pd.Index,
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


def _get_df_from_adata(
    adata: ad.AnnData,
    layer: str | None = None,
) -> pd.DataFrame:
    """Extract dataframe from AnnData object, either from X or a layer.

    Parameters
    ----------
    adata : ad.AnnData
            Anndata object to extract data from.
    layer : str, optional
            Name of the layer to extract. If None, the data matrix X is extracted.

    Returns
    -------
    df : pd.DataFrame
            Dataframe with data from adata.

    """
    if layer is not None:
        return pd.DataFrame(adata.layers[layer], index=adata.obs.index, columns=adata.var.index)
    return pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)


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
