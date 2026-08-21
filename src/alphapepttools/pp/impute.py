# Imputation methods for proteomics data

import logging
from typing import Literal, cast

import anndata as ad
import numpy as np
import pandas as pd
from bpca import BPCA
from sklearn.impute import KNNImputer

from alphapepttools._utils import get_matrix

from ._utils import _is_data_complete, _raise_on_nan_values

logger = logging.getLogger(__name__)


def _impute_gaussian(
    data: np.ndarray,
    std_offset: float = 1.8,
    std_factor: float = 0.3,
    random_state: int = 42,
) -> np.ndarray:
    """Impute missing values in each column by random sampling from a gaussian distribution.

    The distribution is centered at std_offset * feature standard deviation below the
    feature mean and has a standard deviation of std_factor * feature standard deviation.

    The default values are set to mirror Perseus-style imputation: multiply the feature's
    standard deviation by 0.3 and shift the mean down by 1.8 standard deviations, then sample
    from the resulting distribution.

    Parameters
    ----------
    data
        Samples x Features array
    std_offset
        Number of standard deviations below the mean to center the
        gaussian distribution.
    std_factor
        Factor to multiply the feature's standard deviation with to
        get the standard deviation of the gaussian distribution.
    random_state
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Imputed data array
    """
    if _is_data_complete(data):
        logger.info("Data contains no missing values. Skipping imputation.")
        return data

    rng = np.random.default_rng(random_state)

    # generate corresponding downshifted features
    stds = np.nanstd(data, axis=0)
    means = np.nanmean(data, axis=0)
    shifted_means = means - std_offset * stds
    shifted_stds = stds * std_factor

    # iterate over nan-containing columns and impute from corresponding gaussian
    na_col_idxs = np.where(np.isnan(data).sum(axis=0) > 0)[0]
    for i in na_col_idxs:
        na_row_idxs = np.where(np.isnan(data[:, i]))[0]
        data[na_row_idxs, i] = rng.normal(shifted_means[i], shifted_stds[i], len(na_row_idxs))

    return data


def impute_gaussian(
    adata: ad.AnnData,
    group_column: str | None = None,
    layer: str | None = None,
    std_offset: float = 1.8,
    std_factor: float = 0.3,
    random_state: int = 42,
    *,
    copy: bool = False,
) -> ad.AnnData | None:
    """Impute missing values in each column by random sampling from a gaussian distribution.

    The distribution is centered at std_offset * feature standard deviation below the
    feature mean and has a standard deviation of std_factor * feature standard deviation.
    Can perform global imputation using all samples or group-wise imputation
    using subsets of samples defined by a categorical variable.

    Parameters
    ----------
    adata
        AnnData object containing the data to be imputed.
    group_column
        Column name in `adata.obs` defining groups for group-wise imputation.
        If `None` (default), computes statistics across all samples.
        If specified, computes statistics separately for each group and imputes
        missing values using the group-specific gaussian distribution.
        If `group_column` contains NaNs, the respective observations are ignored.
    layer
        Name of the layer to impute. If None (default), the data matrix X is used.
    std_offset
        Number of standard deviations below the mean to center the
        gaussian distribution.
    std_factor
        Factor to multiply the feature's standard deviation with to
        get the standard deviation of the gaussian distribution.
    random_state
        Random seed for reproducibility
    copy
        Whether to return a modified copy (True) of the anndata object. If False (default)
        modifies the object inplace

    Returns
    -------
    None | anndata.AnnData
        AnnData object with imputed values in layer.
        If `copy=False` modifies the anndata object at layer inplace and returns None. If `copy=True`,
        returns a modified copy.

    Raises
    ------
    ValueError
        If `group_column` contains NaNs
    ValueError
        If a feature contains only NaNs

    Notes
    -----
    Features that are fully missing will not be imputed. Appropriate filtering of features with
    :func:`at.pp.filter_data_completeness` is critical.

    Example
    -------
    Impute the values in the `.X` matrix

    .. code-block:: python

        adata = at.pp.impute_gaussian(adata)
        assert np.sum(np.isnan(adata.X)) == 0

    Impute data in a specific layer

    .. code-block:: python

        adata = at.pp.impute_gaussian(adata, layer="layer2")
        assert np.sum(np.isnan(adata.layers["layer2"])) == 0

    Impute groupwise based on a categorical column:

    .. code-block:: python

        adata = at.pp.impute_gaussian(adata, group_column="cell_type")
        # Imputes group-wise gaussian distributions
    """
    adata = adata.copy() if copy else adata

    data = get_matrix(adata, layer)

    if group_column is None:
        _raise_on_nan_values(data, mode="all")
        data = _impute_gaussian(data, std_offset=std_offset, std_factor=std_factor, random_state=random_state)
    else:
        _raise_on_nan_values(
            cast("pd.Series", adata.obs[group_column]),
            mode="any",
            custom_message=f"`group_column` {group_column} contains nans. Cannot impute groups with missing values, please drop these observations prior to imputation.",
        )
        groups = adata.obs.groupby(group_column, dropna=True).indices

        for group_indices in groups.values():
            group = data[group_indices]
            _raise_on_nan_values(group, mode="all")
            data[group_indices, :] = _impute_gaussian(
                group, std_offset=std_offset, std_factor=std_factor, random_state=random_state
            )

    if layer is None:
        adata.X = data
    else:
        adata.layers[layer] = data

    return adata if copy else None


def _impute_nanmedian(data: np.ndarray) -> np.ndarray:
    """Impute nan values in array with column-wise nanmedian

    Parameters
    ----------
    data
        Samples x Features array
    """
    if _is_data_complete(data):
        logger.info("Data contains no missing values. Skipping imputation.")
        return data

    return np.where(np.isnan(data), np.nanmedian(data, axis=0), data)


def impute_median(
    adata: ad.AnnData, group_column: str | None = None, layer: str | None = None, *, copy: bool = False
) -> ad.AnnData | None:
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
    copy
        Whether to return a modified copy (True) of the anndata object. If False (default)
        modifies the object inplace

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
    adata = adata.copy() if copy else adata

    data = get_matrix(adata, layer)

    if group_column is None:
        _raise_on_nan_values(data, mode="all")
        data = _impute_nanmedian(data)
    else:
        if pd.isna(adata.obs[group_column]).any():
            raise ValueError(
                f"`group_column` {group_column} contains nans. The respective observations will be dropped and not get imputed.",
            )

        groups = adata.obs.groupby(group_column, dropna=True).indices

        for group_indices in groups.values():
            group = data[group_indices]
            _raise_on_nan_values(group, mode="all")
            data[group_indices, :] = _impute_nanmedian(group)

    if layer is None:
        adata.X = data
    else:
        adata.layers[layer] = data

    return adata if copy else None


def _impute_knn(data: np.ndarray, **kwargs) -> np.ndarray:
    """Impute missing values using kNN imputation.

    Parameters
    ----------
    data
        Samples x Features array with missing values to impute.
    **kwargs
        Additional keyword arguments passed to sklearn.impute.KNNImputer.

    Returns
    -------
    np.ndarray
        Data array with missing values imputed using k-nearest neighbors.
    """
    if _is_data_complete(data):
        logger.info("Data contains no missing values. Skipping imputation.")
        return data

    imputer = KNNImputer(**kwargs)
    return imputer.fit_transform(data)


def _validate_knn_grouping(groups: dict, n_neighbors: int) -> None:
    """Validate that knn grouping is valid.

    Parameters
    ----------
    groups
        Dictionary mapping group names to indices of samples in each group.
    n_neighbors
        Number of neighbors required for kNN imputation.

    Raises
    ------
    ValueError
        If group keys contain NaN values or if any group has fewer members than n_neighbors.
    """
    if any(pd.isna(key) for key in groups):
        raise ValueError(
            "`group_column` contains nans. The respective observations will be dropped and not get imputed.",
        )

    if any(len(indices) < n_neighbors for _, indices in groups.items()):
        raise ValueError("Number of members per group must be greater equal number of `n_neighbors` for all groups.")


def impute_knn(
    adata: ad.AnnData,
    group_column: str | None = None,
    layer: str | None = None,
    n_neighbors: int = 2,
    weights: Literal["distance", "uniform"] = "distance",
    *,
    copy: bool = False,
    **kwargs,
) -> ad.AnnData | None:
    """Impute missing values using median imputation

    Replace missing (NaN) values for each feature in the data matrix with the estimate based on non-missing
    values in the k nearest observations. Can perform global imputation using all samples or group-wise imputation
    using subsets of samples defined by a categorical variable.

    Parameters
    ----------
    adata
        AnnData object
    layer
        Layer to use for imputation
    group_column
        Column name in `adata.obs` defining groups for group-wise imputation.
            - `None` (default), imputes all samples.
            - `str` Computes median separately for each group
        If `group_column` contains NaNs, the respective observations are ignored.
    n_neighbors
        Number of neighbors to consider during imputation
    weights
        Weighting strategy for kNN imputation.
            - uniform: All k-nearest neighbors are weighted equally for feature imputation
            - distance: The k-nearest neighbors are weighted based on their inverse distance to the imputed observation
    copy
        Whether to return a modified copy (True) of the anndata object. If False (default)
        modifies the object inplace
    **kwargs
        Passed to :class:`sklearn.impute.KNNImputer`

    Returns
    -------
    None | anndata.AnnData
        AnnData object with imputed values in layer.
        If `copy=False` modifies the anndata object at layer inplace and returns None. If `copy=True`,
        returns a modified copy.

    Raises
    ------
    Warning
        If `group_column` contains NaNs
    Warning
        If a feature contains only NaNs
    ValueError
        If any group has less members than `n_neighbors`

    Notes
    -----
    Features that are fully missing will not be imputed. Appropriate filtering of features with
    :func:`at.pp.filter_data_completeness` is critical.
    Nearest neighbors imputation assumes that the data is missing at random. This means that it is
    not appropriate for values that are missing not at random, e.g. due to insufficient instrument sensitivity.
    In this case, kNN imputation will systematically overestimate the intensities of the features.

    Example
    -------
    Impute the values in the `.X` matrix

    .. code-block:: python

        adata = at.pp.impute_knn(adata)
        assert np.sum(np.isnan(adata.X)) == 0

    Impute data in a specific layer

    .. code-block:: python

        adata = at.pp.impute_knn(adata, layer="layer2")
        assert np.sum(np.isnan(adata.layers["layer2"])) == 0

    Impute group-wise based on a categorical column:

    .. code-block:: python

        adata = at.pp.impute_knn(adata, group_column="cell_type")
        # Group-wise imputation
    """
    adata = adata.copy() if copy else adata

    data = get_matrix(adata, layer)

    if group_column is None:
        _raise_on_nan_values(data, mode="all")
        data = _impute_knn(data, n_neighbors=n_neighbors, weights=weights, **kwargs)
    else:
        groups = adata.obs.groupby(group_column, dropna=True).indices
        _validate_knn_grouping(groups=groups, n_neighbors=n_neighbors)

        for group_indices in groups.values():
            group = data[group_indices]
            _raise_on_nan_values(group, mode="all")
            data[group_indices, :] = _impute_knn(group, n_neighbors=n_neighbors, weights=weights, **kwargs)

    if layer is None:
        adata.X = data
    else:
        adata.layers[layer] = data

    return adata if copy else None


def _impute_bpca(data: np.ndarray, n_components: int, **kwargs) -> np.ndarray:
    r"""Imputes missing data with Bayesian Principal Component Analysis

    .. :math:

        X_{estimated} = \approx U\times L + \overline{X}

    References
    ----------
    - :cite:p:`Stacklies.2007`
    """
    bpca = BPCA(n_components=n_components, **kwargs)

    usage = bpca.fit_transform(data)
    loadings = bpca.components_
    mean = bpca.mu

    X_reconstructed = usage @ loadings + mean

    return np.where(np.isnan(data), X_reconstructed, data)


def impute_bpca(
    adata: ad.AnnData,
    *,
    n_components: int = 50,
    layer: str | None = None,
    group_column: str | None = None,
    copy: bool = False,
    **kwargs,
) -> ad.AnnData | None:
    r"""Impute missing values using Bayesian Principal Component Analysis (BPCA)

    Estimates the latent covariance structure of the log-transformed data via Bayesian Principal Component Analysis. The imputation method uses the obtained
    principal component usages :math:`U` and components :math:`L` to reconstruct the full log-transformed data matrix

    .. :math:

        X_{estimated} = \approx U\times L + \overline{X}

    Where :math:`\overline{X}` is the mean of the data. Missing values are replaced with the estimated values from
    this procedure.

    Parameters
    ----------
    adata
        AnnData object.
    n_components
        Number of components to use for the model fit. The more components are used, the more granular the model
        fits the data. This might increase model accuracy but also propagates more measurement noise in the
        data reconstruction.
    layer
        Layer to use for imputation. The data should be log transformed to match the noise model of the BPCA method.
        If `None`, uses the `adata.X` attribute.
    group_column
        Column name in `adata.obs` defining groups for group-wise imputation.
            - `None` (default, recommended), imputes all samples.
            - `str` Computes median separately for each group
        If `group_column` contains NaNs, the respective observations are ignored.
    copy
        Whether to return a modified copy (True) of the anndata object. If False (default)
        modifies the object inplace
    **kwargs
        Passed to :class:`bpca.BPCA`

    Returns
    -------
    None | anndata.AnnData
        AnnData object with imputed values in layer.
        If `copy=False` modifies the anndata object at layer inplace and returns None. If `copy=True`,
        returns a modified copy.

    Raises
    ------
    Warning
        If `group_column` contains NaNs
    Warning
        If a feature contains only NaNs

    Example
    -------

    .. code-block:: python

        # Log transform data
        at.pp.nanlog(adata)

        # Imputes .X inplace
        at.pp.impute_bpca(adata, n_components=50, layer=None)

        # Returns a new anndata object with imputed .X layer
        adata_new = at.pp.impute_bpca(adata, n_components=50, layer=None, copy=True)

    References
    ----------
    This implementation follows the reference implementation in :cite:p:`Stacklies.2007`

    See Also
    --------
    :class:`bpca.BPCA`
    """
    adata = adata.copy() if copy else adata

    data = get_matrix(adata, layer)

    if group_column is None:
        _raise_on_nan_values(data, mode="all")
        data = _impute_bpca(data, n_components=n_components, **kwargs)
    else:
        groups = adata.obs.groupby(group_column, dropna=True).indices

        for group_indices in groups.values():
            group = data[group_indices]
            _raise_on_nan_values(group, mode="all")
            data[group_indices, :] = _impute_bpca(group, n_components=n_components, **kwargs)

    if layer is None:
        adata.X = data
    else:
        adata.layers[layer] = data

    return adata if copy else None
