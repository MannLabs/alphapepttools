# Imputation methods for proteomics data

import logging

import anndata as ad
import numpy as np
import pandas as pd

# logging configuration
logging.basicConfig(level=logging.INFO)


def impute() -> None:
    """Impute missing values in the data"""
    raise NotImplementedError


def impute_gaussian(
    adata: ad.AnnData,
    std_offset: float = 3,
    std_factor: float = 0.3,
    random_state: int = 42,
) -> ad.AnnData:
    """Impute missing values in each column by random sampling from a gaussian distribution.

    The distribution is centered at std_offset * feature standard deviation below the
    feature mean and has a standard deviation of std_factor * feature standard deviation.
    The function returns a copy of the AnnData object with imputed values in place of NaNs.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the data to be imputed.
    std_offset : float
        Number of standard deviations below the mean to center the
        gaussian distribution.
    std_factor : float
        Factor to multiply the feature's standard deviation with to
        get the standard deviation of the gaussian distribution.

    Returns
    -------
    anndata.AnnData
        Copy of AnnData object with imputed values in place of NaNs.

    """
    # always copy for now, implement inplace later if needed
    input_X_shape = adata.X.shape
    adata = adata.copy()
    X = adata.X

    # All columns must be either int or float
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("adata.X must be numeric.")

    nan_count = np.isnan(X).sum()

    # Get the indices of those columns that have missing values: we are going to need downshifted Gaussian's for those
    rng = np.random.default_rng(random_state)
    na_col_idxs = np.where(np.isnan(X).sum(axis=0) > 0)[0]

    if len(na_col_idxs) == 0:
        logging.info(" impute_gaussian: No NaN values found, no imputation performed.")
        return adata

    # generate corresponding downshifted features
    stds = np.nanstd(X, axis=0)
    means = np.nanmean(X, axis=0)
    shifted_means = means - std_offset * stds
    shifted_stds = stds * std_factor

    # iterate over nan-containing columns and impute from corresponding gaussian
    for i in na_col_idxs:
        na_row_idxs = np.where(np.isnan(X[:, i]))[0]
        X[na_row_idxs, i] = rng.normal(shifted_means[i], shifted_stds[i], len(na_row_idxs))

    if not X.shape == input_X_shape:
        raise ValueError(" impute_gaussian: Imputed data shape does not match original data shape.")

    if np.isnan(X).any():
        raise ValueError(" impute_gaussian: Imputation failed, data retained NaN values.")

    logging.info(f" impute_gaussian: Imputation complete. Imputed {nan_count} NaN values with Gaussian distribution.")
    return adata


def _check_all_nan(data: np.ndarray) -> None:
    """Check if a feature contains all nan

    Parameters
    ----------
    data
        Samples x Features array

    Raises
    ------
    ValueError
        If any feature contains only NaNs
    """
    all_nan_features = np.isnan(data).all(axis=0)
    if any(all_nan_features):
        raise ValueError(
            f"Features with index {list(np.where(all_nan_features)[0])} contain all nan values. Drop these features beforehand."
        )


def _impute_nanmedian(data: np.ndarray) -> np.ndarray:
    """Impute nan values in array with column-wise nanmedian

    Parameters
    ----------
    data
        Samples x Features array
    """
    return np.where(np.isnan(data), np.nanmedian(data, axis=0), data)


def impute_median(adata: ad.AnnData, layer: str | None = None, group_column: str | None = None) -> ad.AnnData:
    """Impute missing values using median imputation

    Replace missing (NaN) values in the data matrix with the median of non-missing
    values for each feature. Can perform global imputation using all samples or group-wise imputation
    using subsets of samples defined by a categorical variable.

    Parameters
    ----------
    adata
        AnnData object
    layer
        Layer to use for imputation
    group_column
        Column name in `adata.obs` defining groups for group-wise imputation.
        If `None` (default), computes median across all samples.
        Defines a group column that is used to subset the samples that should be used for imputation.
        If specified, computes median separately for each group and imputes
        missing values using the group-specific median.
        If `group_column` contains NaNs, the respective observations are ignored.

    Returns
    -------
    :class:`ad.AnnData`
        Copy of anndata object with modified layer

    Raises
    ------
    Warning
        If `group_column` contains NaNs
    Warning
        If a feature contains only NaNs

    Notes
    -----
    Features that are fully missing will not be imputed. Appropriate filtering of features with
    :func:`at.pp.filter_data_completeness` is critical.

    Example
    -------
    Impute the values in the `.X` matrix

    .. code-block:: python
        adata = at.pp.impute_median(adata)
        assert np.sum(np.isnan(adata.X)) == 0

    Impute data in a specific layer

    .. code-block:: python
        adata = at.pp.impute_median(adata, layer="layer2")
        assert np.sum(np.isnan(adata.layers["layer2"])) == 0

    Impute groupwise based on a categorical column:

    .. code-block:: python
        adata = at.pp.impute_median(adata, group_column="cell_type")
        # Imputes group-wise medians
    """
    adata = adata.copy()

    data = adata.X if layer is None else adata.layers[layer]

    if group_column is None:
        _check_all_nan(data)
        data = _impute_nanmedian(data)
    else:
        if pd.isna(adata.obs[group_column]).any():
            raise ValueError(
                f"`group_column` {group_column} contains nans. The respective observations will be dropped and not get imputed.",
            )

        groups = adata.obs.groupby(group_column, dropna=True).indices

        for group_indices in groups.values():
            group = data[group_indices]
            _check_all_nan(group)
            data[group_indices, :] = _impute_nanmedian(group)

    if layer is None:
        adata.X = data
    else:
        adata.layers[layer] = data

    return adata
