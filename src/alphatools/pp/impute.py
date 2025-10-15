# Imputation methods for proteomics data

import logging

import anndata as ad
import numpy as np

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
    layer: str | None = None,
    *,
    copy: bool = False,
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
    layer
        Name of the layer to impute. If None (default), the data matrix X is used.
    copy
        Whether to return a modified copy (True) of the anndata object. If False (default)
        modifies the object inplace

    Returns
    -------
    None | anndata.AnnData
        AnnData object with imputed values in layer.
        If `copy=False` modifies the anndata object at layer inplace and returns None. If `copy=True`,
        returns a modified copy.

    """
    # always copy for now, implement inplace later if needed
    adata = adata.copy() if copy else adata

    X = adata.X if layer is None else adata.layers[layer]
    input_X_shape = X.shape

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

    if layer is None:
        adata.X = X
    else:
        adata.layers[layer] = X

    return adata if copy else None
