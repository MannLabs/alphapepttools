# Create and manipulate AnnData objects

import logging
import numbers
import warnings
from typing import Literal

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
    data
        The data array, where rows correspond to samples and columns correspond to features.
        If data is a pd.DataFrame, row and column indices are used for inner join with obs and var.
        If data is a np.ndarray, obs and var are not assigned

    Returns
    -------
    AnnData object with data, obs, and var

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from alphapepttools.pp.data import _to_anndata

        # From numpy array
        data_array = np.random.randn(5, 3)
        adata = _to_anndata(data_array)
        print(adata.shape)  # (5, 3)

        # From DataFrame with labeled indices
        df = pd.DataFrame(
            np.random.randn(4, 3),
            index=["sample1", "sample2", "sample3", "sample4"],
            columns=["protein1", "protein2", "protein3"],
        )
        adata = _to_anndata(df)
        print(adata.obs.index.tolist())  # ['sample1', 'sample2', 'sample3', 'sample4']
        print(adata.var.index.tolist())  # ['protein1', 'protein2', 'protein3']

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
    adata
        AnnData object to add metadata to
    incoming_metadata
        Metadata dataframe to add. The matching entity is always the INDEX, depending on axis it is
        matched against obs (axis = 0) or var (axis = 1)
    axis
        Axis to add metadata to. 0 for obs and 1 for var
    keep_data_shape
        If True, the incoming data is left-joined to the existing data, which may result in nan-padded
        rows in the incoming data. If False, incoming data is added via inner join, which may change the
        shape of the adata object
    keep_existing_metadata
        If True, incoming metadata is added to the existing metadata. If False, incoming metadata replaces
        existing metadata. If columns between existing and incoming metadata are synonymous, the corresponding
        incoming metadata columns are ignored
    verbose
        If True, print additional information about the operation

    Returns
    -------
    AnnData object with metadata added

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import anndata as ad
        import pandas as pd
        from alphapepttools.pp.data import add_metadata

        # Create example AnnData
        adata = ad.AnnData(
            X=np.array([[1, 2], [5, 1], [6, 6], [9, 3]]),
            obs=pd.DataFrame({"batch": ["A", "A", "B", "B"]}, index=["sample1", "sample2", "sample3", "sample4"]),
            var=pd.DataFrame(index=["protein1", "protein2"]),
        )

        # New metadata to add
        new_metadata = pd.DataFrame(
            {"condition": ["control", "control", "treated", "treated"]},
            index=["sample1", "sample2", "sample3", "sample4"],
        )

        # Method 1: Replace existing metadata (keep_existing_metadata=False, default)
        adata1 = adata.copy()
        adata1 = add_metadata(adata1, new_metadata, axis=0)
        print(adata1.obs.columns.tolist())  # ['condition'] - batch is replaced

        # Method 2: Add to existing metadata (keep_existing_metadata=True)
        adata2 = adata.copy()
        adata2 = add_metadata(adata2, new_metadata, axis=0, keep_existing_metadata=True)
        print(adata2.obs.columns.tolist())  # ['batch', 'condition'] - both preserved

    """
    # Basic checks
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1.")

    if adata.shape == (0, 0):
        logging.info("adata is empty")
        return adata

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
    data
        Dataframe to filter. Columns must match with filter_dict keys
    filter_dict
        Dictionary with column names as keys and filter values as values.
        Values can be either string, list or tuple. For strings, exact matches
        are performed. For lists, matches are performed on any element in the
        list. Tuples specify value ranges and must consist of numeric values,
        where 'None' is interpreted as an open end. Ranges are inclusive on
        the lower end and exclusive on the upper end to prevent double counting
        with adjacent filters
    logic
        Filtering logic to apply in case of multiple filters. Default to 'and'.
        Can be 'and' or 'or'

    Returns
    -------
    Boolean mask to filter the dataframe

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
        # TODO: add function evaluation here if v is a callable, so that users can pass things like np.isnan or lambdas
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
    """Tuple-based filtering of numeric features

    Parameters
    ----------
    feature
        Series to filter, must be numeric
    input_tuple
        Tuple of (lower, upper) bounds for filtering. None values are treated as open bounds

    Returns
    -------
    Boolean mask for filtering

    """
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
    """Validate filter dictionary against data

    Parameters
    ----------
    filter_dict
        Dictionary with column names as keys and filter values as values
    data
        DataFrame to validate filter_dict against

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If filter_dict contains invalid keys or values

    """
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
    adata
        AnnData object to filter
    filter_dict
        Dictionary with column names as keys and filter values as values.
        Values can be either string, list or tuple. For strings, exact matches
        are performed. For lists, matches are performed on any element in the
        list. Tuples specify value ranges and must consist of numeric values,
        where 'None' is interpreted as an open end. Ranges are inclusive on
        the lower end and exclusive on the upper end to prevent double counting
        with adjacent filters
    axis
        Axis to filter on. 0 for obs and 1 for var
    logic
        Filtering logic to apply in case of multiple filters. Default to 'and'.
        Can be 'and' or 'or'
    action
        If "keep", extract rows/columns that match the filter conditions.
        If "drop", extract rows/columns outside the filter conditions.
        Default to "keep"

    Returns
    -------
    Filtered anndata object

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import pandas as pd
        import anndata as ad
        import alphapepttools as at

        # Create test data
        X = pd.DataFrame({f"gene_{i}": np.random.randn(6) for i in range(5)}, index=[f"cell_{i}" for i in range(6)])

        sample_metadata = pd.DataFrame(
            {"column1": ["A", "B", "C", "D", "E", "F"], "column2": [50, 200, 50, 200, 50, 200]},
            index=[f"cell_{i}" for i in range(6)],
        )

        test_adata = ad.AnnData(X, obs=sample_metadata)

        # Filter: keep cells with column1 in ["A", "B", "C"] OR column2 between 20-100
        adata_filtered = at.pp.filter_by_metadata(
            test_adata, {"column1": ["A", "B", "C"], "column2": (20, 100)}, axis=0, logic="or", action="keep"
        )
        print(adata_filtered.shape)  # (5, 5) - cells 0,1,2,4 match the criteria

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
    """Check if any fields overlap between two dataframes on respective axes.

    Parameters
    ----------
    data
        DataFrame to check.
    metadata
        Metadata DataFrame to check against.
    axis
        Axis to check (0 for rows, 1 for columns).
    """
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
    """Drop overlapping fields from incoming metadata to avoid name collisions.

    Parameters
    ----------
    metadata
        Incoming metadata DataFrame.
    _inplace_metadata
        Existing metadata DataFrame to check against.
    verbose
        Whether to display warnings about dropped columns.

    Returns
    -------
    pd.DataFrame
        Metadata with overlapping columns removed.
    """
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
    """Get a column from a DataFrame or an AnnData object.

    Parameters
    ----------
    data
        Data to extract the column from.
    column : str
        Column to extract. If data is of type ad.AnnData, the hierarchy to match fields is
        - var_names first
        - obs columns
        - var columns
        If the column is not found in either, a ValueError is raised.

    Returns
    -------
    np.ndarray
        The column data as a numpy array

    Examples
    --------
    Extract from DataFrame:

    .. code-block:: python

        import pandas as pd
        import numpy as np
        from alphapepttools.pp.data import data_column_to_array

        df = pd.DataFrame({"intensity": [100, 200, 150], "protein_id": ["P1", "P2", "P3"]})
        arr = data_column_to_array(df, "intensity")
        print(arr)  # [100 200 150]

    Extract from AnnData:

    .. code-block:: python

        import anndata as ad
        import pandas as pd
        import numpy as np
        from alphapepttools.pp.data import data_column_to_array

        adata = ad.AnnData(
            X=np.array([[1, 2], [3, 4], [5, 6]]),
            obs=pd.DataFrame({"batch": ["A", "A", "B"]}),
            var=pd.DataFrame(index=["protein1", "protein2"]),
        )
        # Get feature data from X
        arr = data_column_to_array(adata, "protein1")
        print(arr)  # [1 3 5]

        # Get metadata from obs
        arr = data_column_to_array(adata, "batch")
        print(arr)  # ['A' 'A' 'B']

    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column {column} not found in DataFrame.")

        return data[column].to_numpy()

    if isinstance(data, ad.AnnData):
        if column in data.var_names:
            col_idx = data.var_names.get_loc(column)
            return data.X[:, col_idx].flatten()

        if column in data.obs.columns:
            return data.obs[column].to_numpy()

        if column in data.var.columns:
            return data.var[column].to_numpy()

        raise ValueError(
            f"Column {column} not found in AnnData object (checked var_names and obs.columns and var.columns)."
        )

    raise TypeError(f"Expected pd.DataFrame or ad.AnnData, got {type(data)}")


def data_index_to_array(
    data: pd.DataFrame | ad.AnnData,
    dim_space: str = "obs",
) -> np.ndarray:
    """Get indices from a DataFrame or an AnnData object

    Parameters
    ----------
    data : pd.DataFrame | ad.AnnData
        Data to extract indices from.
    dim_space : str, optional
        Dimension space to extract indices from. Either "obs" for observation/row indices
        or "var" for variable/column indices. By default "obs".

    Returns
    -------
    np.ndarray
        The indices as a numpy array

    Raises
    ------
    ValueError
        If dim_space is not "obs" or "var"
    TypeError
        If data is not a DataFrame or AnnData object

    Examples
    --------
    >>> import pandas as pd
    >>> import anndata as ad
    >>> import numpy as np
    >>>
    >>> # DataFrame example
    >>> df = pd.DataFrame({"A": [1, 2, 3]}, index=["row1", "row2", "row3"])
    >>> data_index_to_array(df, "obs")
    array(['row1', 'row2', 'row3'], dtype=object)
    >>> data_index_to_array(df, "var")
    array(['A'], dtype=object)
    >>>
    >>> # AnnData example
    >>> adata = ad.AnnData(np.random.rand(3, 2))
    >>> adata.obs_names = ["cell1", "cell2", "cell3"]
    >>> adata.var_names = ["gene1", "gene2"]
    >>> data_index_to_array(adata, "obs")
    array(['cell1', 'cell2', 'cell3'], dtype=object)
    >>> data_index_to_array(adata, "var")
    array(['gene1', 'gene2'], dtype=object)
    """
    if dim_space not in ["obs", "var"]:
        raise ValueError(f"dim_space must be 'obs' or 'var', got '{dim_space}'")

    if isinstance(data, pd.DataFrame):
        if dim_space == "obs":
            return data.index.to_numpy()
        # dim_space == "var"
        return data.columns.to_numpy()

    if isinstance(data, ad.AnnData):
        if dim_space == "obs":
            return data.obs_names.to_numpy()
        # dim_space == "var"
        return data.var_names.to_numpy()

    raise TypeError(f"Expected pd.DataFrame or ad.AnnData, got {type(data)}")


def _tolist(
    obj: str | list,
) -> list:
    return obj if isinstance(obj, list) else [obj]


def subset_data(
    data: pd.DataFrame | ad.AnnData,
    idxs: pd.Index | list[int],
) -> pd.DataFrame | ad.AnnData:
    """Subset data based on indices

    filtering data based on provided indices, handle both pd.DataFrame and AnnData objects.
    The returned object is a copy of the input data with only the specified indices.

    Parameters
    ----------
    data : pd.DataFrame | ad.AnnData
        Data to subset.
    idxs : pd.Index | list[int]
        Indices to subset the data.

    Returns
    -------
    pd.DataFrame | ad.AnnData
        Subsetted data.

    """
    if isinstance(data, pd.DataFrame):
        return data.iloc[idxs].copy()
    # AnnData
    return data[idxs].copy()


def coerce_to_dataframe(
    data: pd.DataFrame | ad.AnnData,
) -> pd.DataFrame:
    """Coerce data to a DataFrame

    If data is already a DataFrame, it is returned as is. If data is an AnnData object, the default .to_df() method is used to convert the X matrix to a DataFrame with obs index and var names as columns.

    Parameters
    ----------
    data : pd.DataFrame | ad.AnnData
        Data to coerce.

    Returns
    -------
    pd.DataFrame
        Coerced DataFrame.

    """
    if isinstance(data, pd.DataFrame):
        return data
    # AnnData case: convert X to DataFrame with obs index and var names as columns
    return pd.concat([data.to_df(), data.obs], axis=1)


def data_columns_to_df(
    data: ad.AnnData | pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Extract selected columns from AnnData or DataFrame

    Adapter function for matplotlib plotting that extracts columns from either
    AnnData's X matrix/obs or from a DataFrame. Validates there are no duplicates
    when extracting from AnnData.

    Parameters
    ----------
    data
        Input data object (AnnData or DataFrame)
    columns
        Column names to extract. If `None`, returns all columns (DataFrame)
        or all columns in X (AnnData)

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame containing the selected columns

    Examples
    --------
    Extract from DataFrame:

    .. code-block:: python

        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = data_columns_to_df(df, columns=["A"])

    Extract from AnnData (searches both X and obs):

    .. code-block:: python

        # Get protein intensities from X and sample metadata from obs
        df = data_columns_to_df(adata, columns=["Protein1", "sample_type"])

    Use in plotting:

    .. code-block:: python

        # Prepare data for violin plot
        df = data_columns_to_df(adata, columns=["group", "intensity"])
        ax.violinplot(df.groupby("group")["intensity"].apply(list))
    """
    # Prevent double extraction of the same column
    if columns is not None:
        columns = pd.Series(columns).drop_duplicates().tolist()

    if isinstance(data, pd.DataFrame):
        columns = columns or data.columns.tolist()
        try:
            dataset = data[columns]
        except KeyError as e:
            raise KeyError(f"Columns {columns} not found in dataframe.") from e

    elif isinstance(data, ad.AnnData):
        if columns is None:
            dataset = data.to_df()
        else:
            # Partition columns by source
            x_cols = [col for col in columns if col in data.var_names]
            obs_cols = [col for col in columns if col in data.obs.columns]

            # Check for duplicate columns across sources
            duplicates = set(x_cols) & set(obs_cols)
            if duplicates:
                raise KeyError(
                    f"Columns {duplicates} found in both AnnData X and obs. Please ensure unique column names."
                )

            # Check for missing columns
            missing_cols = set(columns) - set(x_cols) - set(obs_cols)
            if missing_cols:
                raise KeyError(f"Columns {missing_cols} not found in AnnData X or obs.")

            # Build dataset from available sources
            parts = []
            if x_cols:
                parts.append(data.to_df()[x_cols])
            if obs_cols:
                parts.append(data.obs[obs_cols])

            dataset = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]

    else:
        raise TypeError(f"Expected pd.DataFrame or ad.AnnData, got {type(data)}")

    return dataset


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
        AnnData object with data to scale.
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

    Examples
    --------
    Apply standard scaling to data:

    .. code-block:: python

        import anndata as ad
        import pandas as pd
        import numpy as np
        from alphapepttools.pp.data import scale_and_center

        adata = ad.AnnData(
            X=np.array([[1, 10], [2, 20], [3, 30], [4, 40]]),
            obs=pd.DataFrame({"sample": ["S1", "S2", "S3", "S4"]}),
            var=pd.DataFrame(index=["protein1", "protein2"]),
        )

        # Standard scaling (in-place)
        scale_and_center(adata, scaler="standard")

        # Robust scaling on a specific layer
        adata.layers["raw"] = adata.X.copy()
        scale_and_center(adata, scaler="robust", layer="raw")

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
def _validate_adata_for_completeness_filter(adata: ad.AnnData, action: str, var_colname: str) -> None:
    """Validate AnnData object for data completeness filtering.

    Checks that the AnnData object meets requirements for completeness filtering:
    - Object is an actual AnnData instance
    - Contains at least one feature (column)
    - Data matrix X contains numeric values only
    - No duplicated indices in obs
    - Action parameter is either 'flag' or 'drop'
    - Warns if var_colname already exists when action is 'flag'

    Parameters
    ----------
    adata
        AnnData object to validate.
    action
        Action to perform ('flag' or 'drop').
    var_colname
        Name of the column in var to flag features.

    Raises
    ------
    TypeError
        If adata is not an AnnData object or X is not numeric.
    ValueError
        If adata has no features, has duplicated indices, or action is invalid.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an AnnData object.")

    if adata.shape[1] == 0:
        raise ValueError("adata has no features (columns).")

    if not is_numeric_dtype(adata.X):
        raise ValueError("adata.X must be numeric.")

    if any(adata.obs.index.duplicated()):
        raise ValueError("pp.filter_data_completeness(): Duplicated indices in obs")

    if action not in ["flag", "drop"]:
        raise ValueError("pp.filter_data_completeness(): action must be 'flag' or 'drop'.")

    if var_colname in adata.var.columns and action == "flag":
        logging.info(
            f"pp.filter_data_completeness(): `adata.var` column '{var_colname}' already exists, will overwrite."
        )


def filter_data_completeness(
    adata: ad.AnnData,
    max_missing: float,
    group_column: str | None = None,
    groups: list[str] | None = None,
    keep_strategy: Literal["any", "all"] = "all",
    action: str = "flag",
    var_colname: str = "passed_threshold_missing_values",
) -> ad.AnnData:
    """Filter features based on missing values.

    Filters AnnData features (columns) based on the fraction of missing values.
    If group_column and groups are provided, only missingness of certain metadata
    levels is considered. This is especially useful for imbalanced classes, where
    filtering by global missingness may leave too many missing values in the smaller
    class.

    (In case rows should be filtered, it is recommended to transpose the adata
    object prior to calling this function and reverting the transpose afterwards.)

    Parameters
    ----------
    adata
        AnnData object
    max_missing
        Maximum fraction of missing values allowed to pass filtering.
    group_column
        Column name in `adata.obs` defining groups for group-wise filtering.
        If `None` (default), computes missingness across all samples.
        If specified, computes statistics separately for each group.
        This is useful to retain features that are exclusive for a specific sample group.
    groups
        List of levels of the group_column to consider in filtering. E.g. if the column has the levels
        ['A', 'B', 'C'], and groups = ['A', 'B'], only missingness of features in these
        groups is considered. If None, all groups are considered.
    keep_strategy
        Only relevant for groupwise filtering
        - `any` : keep a feature if it passes the threshold in at least one group.
        - `all` : keep a feature only if it passes in every group. This is useful in highly unbalanced designs.
    action
        Action to perform. can be 'flag' (default) or 'drop'. If 'flag', a boolean column in `adata.var`
        is added to indicate whether the feature passed the missingness threshold. If 'drop',
        features that do not pass the threshold are dropped from the AnnData object.
    var_colname
        Name of the `adata.var` boolean column to add if action is 'flag'. Default is 'passed_threshold_missing_values'.

    Returns
    -------
    AnnData
        AnnData object with either a new `adata.var` column added (if `flag`)
        or filtered features (if `drop`).

    Examples
    --------
    Flag features with too many missing values:

    .. code-block:: python

        import anndata as ad
        import pandas as pd
        import numpy as np
        from alphapepttools.pp.data import filter_data_completeness

        # Create data with missing values
        X = np.array(
            [[1.0, np.nan, 3.0, 4.0], [2.0, np.nan, 6.0, 8.0], [3.0, 5.0, np.nan, 12.0], [4.0, 6.0, 9.0, 16.0]]
        )
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame({"group": ["A", "A", "B", "B"]}),
            var=pd.DataFrame(index=["prot1", "prot2", "prot3", "prot4"]),
        )

        # Flag features with >30% missing values
        adata = filter_data_completeness(adata, max_missing=0.3, action="flag")

        # Drop features with >30% missing in group A only
        adata = filter_data_completeness(adata, max_missing=0.3, group_column="group", groups=["A"], action="drop")

    Filter by group-specific completeness:

    .. code-block:: python

        # Consider missingness only in specific groups
        adata = filter_data_completeness(adata, max_missing=0.5, group_column="group", groups=["A", "B"], action="flag")

    """
    if max_missing < 0 or max_missing > 1:
        raise ValueError("Threshold must be between 0 and 1.")

    _validate_adata_for_completeness_filter(adata, action, var_colname)

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
    drop_mask = np.array([False] * adata.shape[1])
    for indices in group_indices.values():
        missing_fraction = np.isnan(adata[indices, :].X).mean(axis=0)
        drop_mask |= missing_fraction > max_missing
        n_dropped = drop_mask.sum()

    # depending on action, either flag or drop features
    if action == "drop":
        if drop_mask.any():
            adata = adata[:, ~drop_mask].copy()
    else:
        adata.var[var_colname] = ~drop_mask

    logging.info(
        f"pp.filter_data_completeness(): {action} {n_dropped} / {drop_mask.size} features with >{max_missing:.2f} missing in any group."
    )
    return adata
