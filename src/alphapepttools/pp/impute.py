# Imputation methods for proteomics data

import logging
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)


def _check_for_complete_data(
    data: np.ndarray,
) -> bool:
    """Check if data contains any missing values

    Parameters
    ----------
    data
        Samples x Features array

    Returns
    -------
    bool
        True if data contains no missing values, False otherwise
    """
    return not np.any(np.isnan(data))


def _raise_on_all_nan_values(data: np.ndarray) -> None:
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
    if _check_for_complete_data(data):
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
    std_offset: float = 3,
    std_factor: float = 0.3,
    random_state: int = 42,
    *,
    copy: bool = False,
) -> ad.AnnData:
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

    data = adata.X if layer is None else adata.layers[layer]

    if group_column is None:
        _raise_on_all_nan_values(data)
        data = _impute_gaussian(data, std_offset=std_offset, std_factor=std_factor, random_state=random_state)
    else:
        if pd.isna(adata.obs[group_column]).any():
            raise ValueError(
                f"`group_column` {group_column} contains nans. Cannot impute groups with missing values.",
            )

        groups = adata.obs.groupby(group_column, dropna=True).indices

        for group_indices in groups.values():
            group = data[group_indices]
            _raise_on_all_nan_values(group)
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
    if _check_for_complete_data(data):
        logger.info("Data contains no missing values. Skipping imputation.")
        return data

    return np.where(np.isnan(data), np.nanmedian(data, axis=0), data)


def impute_median(
    adata: ad.AnnData, group_column: str | None = None, layer: str | None = None, *, copy: bool = True
) -> ad.AnnData:
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

    data = adata.X if layer is None else adata.layers[layer]

    if group_column is None:
        _raise_on_all_nan_values(data)
        data = _impute_nanmedian(data)
    else:
        if pd.isna(adata.obs[group_column]).any():
            raise ValueError(
                f"`group_column` {group_column} contains nans. The respective observations will be dropped and not get imputed.",
            )

        groups = adata.obs.groupby(group_column, dropna=True).indices

        for group_indices in groups.values():
            group = data[group_indices]
            _raise_on_all_nan_values(group)
            data[group_indices, :] = _impute_nanmedian(group)

    if layer is None:
        adata.X = data
    else:
        adata.layers[layer] = data

    return adata if copy else None


def _impute_knn(data: np.ndarray, **kwargs) -> np.ndarray:
    """Impute missing values using kNN imputation"""
    if _check_for_complete_data(data):
        logger.info("Data contains no missing values. Skipping imputation.")
        return data

    imputer = KNNImputer(**kwargs)
    return imputer.fit_transform(data)


def _validate_knn_grouping(groups: dict, n_neighbors: int) -> None:
    """Validate that knn grouping is valid"""
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
) -> ad.AnnData:
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

    data = adata.X if layer is None else adata.layers[layer]

    if group_column is None:
        _raise_on_all_nan_values(data)
        data = _impute_knn(data, n_neighbors=n_neighbors, weights=weights, **kwargs)
    else:
        groups = adata.obs.groupby(group_column, dropna=True).indices
        _validate_knn_grouping(groups=groups, n_neighbors=n_neighbors)

        for group_indices in groups.values():
            group = data[group_indices]
            _raise_on_all_nan_values(group)
            data[group_indices, :] = _impute_knn(group, n_neighbors=n_neighbors, weights=weights, **kwargs)

    if layer is None:
        adata.X = data
    else:
        adata.layers[layer] = data

    return adata if copy else None
