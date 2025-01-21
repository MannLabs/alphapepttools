# Create and manipulate Anndata objects

import logging

import anndata as ad
import numpy as np
import pandas as pd
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


def add_metadata(
    adata: ad.AnnData,
    metadata: pd.DataFrame,
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

    if not isinstance(metadata, pd.DataFrame) or metadata.index.nlevels > 1:
        raise TypeError("metadata must be a pd.DataFrame with single-level index.")

    # duplicate metadata indices are not supported
    if any(metadata.index.duplicated()):
        raise ValueError("Duplicated metadata indices are not supported.")

    # set join type
    join = "inner" if not keep_data_shape else "left"

    # if existing metadata should be kept and new metadata contains synonymous fields to existing metadata, drop incoming fields
    if keep_existing_metadata:
        if axis == 0:
            _inplace_metadata = adata.obs
        elif axis == 1:
            _inplace_metadata = adata.var

        # handle overlapping metadata columns
        metadata = _handle_overlapping_columns(metadata, _inplace_metadata, verbose=verbose)

        # join new to existing metadata; same join logic as for data
        if verbose:
            logging.info(f"pp.add_metadata(): Join incoming to existing metadata via {join} join on axis  {axis}.")
        metadata = _inplace_metadata.join(metadata, how=join)

    # align metadata to data. Based on axis, metadata index will be aligned
    # to either data index (axis = 0) or data columns (axis = 1)
    if axis == 0:
        df, md = _get_df_from_adata(adata).align(metadata, axis=0, join=join)
        if not df.index.equals(md.index):
            raise ValueError("Attempted alignment; metadata indices do not match data indices.")
    elif axis == 1:
        # transpose metadata to align with data columns
        df, md = _get_df_from_adata(adata).align(metadata.T, axis=1, join=join)
        if not df.columns.equals(md.columns):
            raise ValueError("Attempted alignment; metadata columns do not match data columns.")

    # new AnnData object
    adata_new = ad.AnnData(df.values)
    adata_new.obs = md if axis == 0 else adata.obs
    adata_new.var = md.T if axis == 1 else adata.var
    if verbose:
        logging.info(
            f"pp.add_metadata(): Data {adata.shape} to {adata_new.shape}; obs {adata.obs.shape} to {adata_new.obs.shape}; var {adata.var.shape} to {adata_new.var.shape} \n"
        )

    return adata_new


def _handle_overlapping_columns(
    metadata: pd.DataFrame,
    _inplace_metadata: pd.DataFrame,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop overlapping fields from incoming metadata to avoid name collisions"""
    overlapping_fields = metadata.columns.intersection(_inplace_metadata.columns)
    if overlapping_fields.size > 0:
        if verbose:
            logging.info(
                f"pp.add_metadata(): Synonymous fields, dropping {overlapping_fields.to_list()} from incoming metadata."
            )
        return metadata.drop(
            overlapping_fields,
            axis=1,
            errors="ignore",
            inplace=False,
        )
    return metadata


def _get_df_from_adata(
    adata: ad.AnnData,
) -> pd.DataFrame:
    """Extract dataframe from AnnData object

    Parameters
    ----------
    adata : ad.AnnData
            Anndata object to extract data from.

    Returns
    -------
    df : pd.DataFrame
            Dataframe with data from adata.

    """
    return pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)


def scale_and_center(  # explicitly tested via test_pp_scale_and_center()
    adata: ad.AnnData,
    scaler: str = "standard",
    to_layer: str | None = None,
    from_layer: str | None = None,
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
    to_layer : str, optional
        Name of the layer to scale. If None, the data matrix X is modified
    from_layer : str, optional
        Name of the layer to scale. If None, the data matrix X is used.

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

    if to_layer is None:
        adata.X = scaler.fit_transform(adata.X if from_layer is None else adata.layers[from_layer])
    else:
        adata.layers[to_layer] = scaler.fit_transform(adata.X if from_layer is None else adata.layers[from_layer])
