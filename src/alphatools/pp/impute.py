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
    """Impute missing values in each column by sampling from a gaussian distribution.

    The distribution is centered at std_offset standard deviations below the mean of
    the feature and has a standard deviation of std_factor times the feature's.

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
        AnnData object with imputed values in place of NaNs.

    """
    # always copy for now, implement inplace later if needed
    adata = adata.copy()
    X = adata.X

    # All columns must be either int or float
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("adata.X must be numeric.")

    # Get the indices of those columns that have missing values: we are going to need downshifted Gaussian's for those
    np.random.seed(random_state)
    na_col_idxs = np.where(np.isnan(X).sum(axis=0) > 0)[0]

    # generate corresponding downshifted features
    stds = np.nanstd(X, axis=0)
    means = np.nanmean(X, axis=0)
    shifted_means = means - std_offset * stds
    shifted_stds = stds * std_factor

    # iterate over nan-containing columns and impute from corresponding gaussian
    for i in na_col_idxs:
        na_row_idxs = np.where(np.isnan(X[:, i]))[0]
        X[na_row_idxs, i] = np.random.normal(shifted_means[i], shifted_stds[i], len(na_row_idxs))

    if not X.shape == adata.X.shape:
        raise ValueError("Imputed data shape does not match original data shape.")

    # set imputed values back to adata
    adata.X = X
    logging.info("Imputation complete.")
    return adata
