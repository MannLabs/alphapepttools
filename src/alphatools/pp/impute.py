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
) -> ad.AnnData:
    """Impute missing values in each column by random sampling from a gaussian distribution.

    The distribution is centered at std_offset * feature standard deviation below the
    feature mean and has a standard deviation of std_factor * feature standard deviation.
    The function returns a copy of the AnnData object with imputed values in place of NaNs.

    If a column is entirely missing, replace nans with 0.

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

    # Mean and std of data columns
    stds = np.nanstd(X, axis=0)
    means = np.nanmean(X, axis=0)

    # replace nans with 0. This means that entirely missing columns will be replaced with 0.
    stds[np.isnan(stds)] = 0
    means[np.isnan(means)] = 0

    # generate corresponding downshifted features
    shifted_means = means - std_offset * stds
    shifted_stds = stds * std_factor

    # iterate over nan-containing columns and impute from corresponding gaussian
    for i in na_col_idxs:
        na_row_idxs = np.where(np.isnan(X[:, i]))[0]
        X[na_row_idxs, i] = rng.normal(shifted_means[i], shifted_stds[i], len(na_row_idxs))

    if not X.shape == input_X_shape:
        raise ValueError("Imputed data shape does not match original data shape.")

    if np.isnan(X).any():
        raise ValueError("Imputation failed, data retained NaN values.")

    logging.info(f"Imputation complete. Imputed {nan_count} NaN values with Gaussian distribution.")
    return adata
