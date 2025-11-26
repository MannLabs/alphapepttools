"""Feature level metrics"""

import anndata as ad
import numpy as np


def _cv(data: np.ndarray, *, min_valid: int = 3, axis: int = 0) -> np.ndarray:
    """Compute the coefficient of variation

    Parameters
    ----------
    data
        Array of shape (observations, features)
    min_valid
        Minimum number of samples with non-na values to compute the value.
    axis
        Axis along which to compute CV (defaults to feature-wise)

    Returns
    -------
    np.ndarray
       1D Array with length of axis with computed CVs and `nan` where the
       number of non-na values is smaller than min_valid.
    """
    cv = np.nanstd(data, axis=axis) / np.nanmean(data, axis=axis)
    valid_count = np.sum(~np.isnan(data), axis=axis)
    cv[valid_count < min_valid] = np.nan
    return cv


def coefficient_of_variation(
    adata: ad.AnnData,
    *,
    min_valid: int = 3,
    key_added: str = "cv",
    layer: str | None = None,
    copy: bool = False,
) -> None | ad.AnnData:
    r"""Coefficient of variation

    Compute the coefficient of variation (CV) for all features.

    .. math::

        CV = \frac{s(X)}{\hat{X}}

    with the empirical standard deviation :math:`s(X)` of feature :math:`X` and the
    empirical mean :math:`\hat{X}`

    The coefficient of variation is a scale-invariant measure of dispersion that enables
    comparison of variability across features with different abundance levels.

    Within technical replicates, the CV indicates measurement reproducibility. Lower CVs
    indicate good technical precision, while high CVs suggest issues
    with sample preparation, instrument performance, or quantification accuracy.

    Between different biological samples, CVs reflect both biological and technical variation.
    Higher CVs are expected and can indicate genuine biological heterogeneity.

    Parameters
    ----------
    adata
        AnnData object
    min_valid
        Minimum number of samples required to estimate the CV. Will be set to `NaN` otherwise.
    key_added
        Name of column added to `adata.var`
    layer
        Name of the layer to compute metric on. If None (default), the data matrix X is used.
    copy
        Whether to return a modified copy (True) of the anndata object. If False (default)
        modifies the object inplace

    Returns
    -------
    None | anndata.AnnData
        AnnData object with imputed values in layer.
        If `copy=False` modifies the anndata object at layer inplace and returns None. If `copy=True`,
        returns a modified copy.

    Notes
    -----
    The CV only considers non-missing values and should be computed before imputation.
    Features with fewer than `min_valid` non-missing values will return NaN for CV.

    """
    adata = adata.copy() if copy else adata
    data = adata.X if layer is None else adata.layers[layer]

    adata.var[key_added] = _cv(data, min_valid=min_valid, axis=0)

    return adata if copy else None
