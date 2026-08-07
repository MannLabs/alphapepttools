import warnings
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import gmean

from ._utils import _raise_on_missing_value, _raise_on_nan_values

STRATEGIES = ["total_mean", "total_median"]


def _validate_strategies(strategy: str) -> None:
    """Verify that valid strategy was selected.

    Parameters
    ----------
    strategy
        Normalization strategy to validate.

    Raises
    ------
    ValueError
        If strategy is not in the list of valid strategies.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"`strategy` must be one of {STRATEGIES}, not {strategy}")


def _total_mean_normalization(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total normalization

    Normalizes total intensity in each sample (row) to mean of the total intensities.
    NaN-values are interpreted as zero-values.

    Parameters
    ----------
    data
        Count data of shape (samples, features)

    Examples
    --------
    Each sample has the same total intensity:

    .. code-block:: python

        arr = np.array([[1, 1], [2, 0], [0, 2]])
        arr_norm, factors = _total_mean_normalization(arr)
        arr_norm
        > array([[1., 1.],
                [2., 0.],
                [0., 2.]])
        (arr_norm == arr).all()
        > True

    Sample 0 has a lower total intensity:

    .. code-block:: python

        arr = np.array([[0.8, 1], [2, 0], [0, 2]])
        arr_norm, factors = _total_mean_normalization(arr)
        arr_norm.sum(axis=1)
        > array([1.93333333, 1.93333333, 1.93333333])
    """
    # Compute sample-wise means
    # NaNs are interpreted as zero-values
    total_counts = np.nansum(data, axis=1)
    norm_factors = np.mean(total_counts) / total_counts

    return data * norm_factors.reshape(-1, 1), norm_factors


def _total_median_normalization(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total normalization

    Normalizes total intensity in each sample (row) to median of the total intensities
    NaN-values are interpreted as zero-values.

    Parameters
    ----------
    data
        Count data of shape (samples, features)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of normalized data and scaling factors

    Examples
    --------
    Each sample has the same total intensity:

    .. code-block:: python

        arr = np.array([[1, 1], [2, 0], [0, 2]])
        arr_norm, factors = _total_median_normalization(arr)
        (arr_norm == arr).all()
        > True

    Sample 0 has a lower total intensity:

    .. code-block:: python

        arr = np.array([[0.8, 1], [2, 0], [0, 3]])
        arr_norm, factors = _total_median_normalization(arr)
        arr_norm.sum(axis=1)
        > array([2., 2., 2.])

    See Also
    --------
    alphapepttools.pp.norm._total_mean_normalization
    """
    # Compute sample-wise means
    # NaNs are counted as zeros
    total_counts = np.nansum(data, axis=1)
    norm_factors = np.median(total_counts) / total_counts

    return data * norm_factors.reshape(-1, 1), norm_factors


def normalize(
    adata: ad.AnnData,
    layer: str | None = None,
    strategy: Literal["total_mean", "total_median"] = "total_mean",
    group_column: str | None = None,
    key_added: str | None = None,
    *,
    copy: bool = False,
) -> ad.AnnData | None:
    """Normalize measured counts per sample

    Parameters
    ----------
    adata
        Count data
    layer:
        Layer that will be normalized. If `None` uses `anndata.AnnData.X`
    strategy
        Normalization strategy

            - *total_mean* The intensity of each feature is adjusted by a normalizing factor so that the
            total sample intensity is equal to the mean of the total sample intensities across all samples
            - *total_median* The intensity of each feature is adjusted by a normalizing factor so that the
            total sample intensity is equal to the median of the total sample intensities across all samples
    group_column
        Column name in `adata.obs` defining groups for group-wise normalization.
        If `None` (default), computes statistics across all samples.
        If specified, computes statistics separately for each group.
        This is useful when working with data from different batches with different intensity distributions.
    key_added
        If not None, adds normalization factors to column in `adata.obs`
    copy
        Whether to return a modified copy (True) of the anndata object. If False (default)
        modifies the object inplace

    Returns
    -------
    None | anndata.AnnData
        AnnData object with normalized measurement layer.
        If `copy=False` modifies the anndata object at layer inplace and returns None. If `copy=True`,
        returns a modified copy.

    Examples
    --------
    Create an AnnData object with intensity data:

    .. code-block:: python

        adata = ad.AnnData(X=np.array([[0.8, 1.0], [2.0, 0.0], [0.0, 2.0]]))
        adata.X
        > array([[0.8, 1.0],
                [2.0, 0.0],
                [0.0, 2.0]])

    The anndata object gets normalized in place. Per default, the `.X` attribute will be modified:

    .. code-block:: python

        normalize(adata)
        adata.X
        > array([[0.85925926, 1.07407407],
                [1.93333333, 0.        ],
                [0.        , 1.93333333]])

    Alternatively, we can normalize a different layer:

    .. code-block:: python

        adata.layers["normalized"] = adata.X.copy()
        normalize(adata, strategy="total_mean", layer="normalized")
        adata.X
        > array([[0.8, 1.0],
                [2.0, 0.0],
                [0.0, 2.0]])
        adata.layers["normalized"]
        > array([[0.85925926, 1.07407407],
                [1.93333333, 0.        ],
                [0.        , 1.93333333]])

    Or we return a copy of the object:

    .. code-block:: python

        new_adata = normalize(adata, copy=True)
    """
    _validate_strategies(strategy=strategy)

    adata = adata.copy() if copy else adata

    data = adata.layers[layer] if layer is not None else adata.X

    norm_func = _total_mean_normalization if strategy == "total_mean" else _total_median_normalization

    if group_column is None:
        normalized_data, norm_factors = norm_func(data)
    else:
        _raise_on_nan_values(
            adata.obs[group_column],
            mode="any",
            custom_message=f"`group_column` {group_column} contains nans. Cannot normalize groups with missing values, please drop these observations prior to normalization.",
        )
        groups = adata.obs.groupby(group_column, dropna=True).indices

        normalized_data = np.empty_like(data)
        norm_factors = np.empty(data.shape[0])
        for group_indices in groups.values():
            group_normalized, group_factors = norm_func(data[group_indices])
            normalized_data[group_indices, :] = group_normalized
            norm_factors[group_indices] = group_factors

    # Reassign to anndata
    if layer is None:
        adata.X = normalized_data
    else:
        adata.layers[layer] = normalized_data

    if key_added is not None:
        adata.obs[key_added] = norm_factors

    return adata if copy else None


def irs(
    adata: ad.AnnData,
    group_column: str,
    reference_column: str | None = None,
    reference_value: object | None = None,
    *,
    layer: str | None = None,
    copy: bool = False,
) -> None | ad.AnnData:
    """Internal Reference Scaling (IRS) normalization.

    Normalize features across multiple runs (e.g. TMT plexes) using a shared
    internal reference, as commonly performed in isobaric labelling experiments
    :cite:`Plubell.2017`. For each run defined by `group_column`, a per-feature
    reference profile is computed from the samples where
    `reference_column == reference_value`. Every sample in that run is then
    rescaled so its reference profile matches the geometric mean of reference
    profiles taken across all runs. NaNs are propagated.

    If `reference_column` is `None`, the per-run arithmetic mean across all
    samples is used in place of an explicit reference.

    Parameters
    ----------
    adata
        AnnData object
    group_column
        Column in `adata.obs` that defines the individual runs.
    reference_column
        Column in `adata.obs` indicating which samples are reference channels.
        If `None`, a virtual reference is constructed from the arithmetic mean
        of all samples within each run.
    reference_value
        Value in `reference_column` that marks the reference sample(s) within
        each run. Ignored when `reference_column` is `None`.
    layer
        Layer in `adata` to normalize. If `None`, uses `adata.X`.
    copy
        Whether to return a modified copy (True) of the anndata object. If False (default)
        modifies the object inplace

    Reference
    ---------
    - Plubell, D. L. et al. Extended Multiplexing of Tandem Mass Tags (TMT) Labeling Reveals Age and High Fat Diet Specific Proteome Changes in Mouse Epididymal Adipose Tissue. Mol Cell Proteomics 16, 873-890 (2017).
    - Phillip Wilmarth. Thorough Testing of Internal Reference Scaling (IRS) [Website]. https://pwilmart.github.io/TMT_analysis_examples/IRS_validation.html. (2019)
    """
    if (reference_column is not None) and (reference_value is None):
        warnings.warn(
            "`reference_value` is None while `reference_column` is set - is this intended?",
            stacklevel=2,
        )
    if (reference_column is None) and (reference_value is not None):
        warnings.warn(
            "`reference_value` is set while `reference_column` is None - it will be ignored.",
            stacklevel=2,
        )

    adata = adata.copy() if copy else adata

    data = adata.layers[layer].copy() if layer is not None else adata.X.copy()

    _raise_on_nan_values(
        adata.obs[group_column],
        mode="any",
        custom_message=f"`group_column` {group_column} contains nans. Cannot normalize groups with missing values, please drop these observations prior to normalization.",
    )

    groups = adata.obs.groupby(group_column, dropna=True)

    sample_ref_values = np.full_like(data, np.nan, dtype=float)
    group_ref_values = np.full(shape=(len(groups), adata.n_vars), fill_value=np.nan, dtype=float)

    # The internal reference value is either computed from the internal reference samples, as indicated by the reference column,
    # or it is computed as the mean of all channels in the respective run.
    for group_idx, (group_name, group_indices) in enumerate(groups.indices.items()):
        if reference_column is not None:
            group_metadata: pd.DataFrame = groups.get_group(group_name)
            _raise_on_missing_value(
                group_metadata[reference_column],
                reference_value,
                value_name="reference_value",
                custom_message=f"Group {group_name!r} does not contain a reference sample.",
            )

            # Estimate reference from group-specific reference samples
            ref_indices = np.where(group_metadata[reference_column] == reference_value)[0]
            ref_data = data[group_indices, :][ref_indices, :]
        else:
            # If no reference value exists, estimate reference from all samples
            ref_data = data[group_indices, :]

        ref_value = np.nanmean(ref_data, axis=0).squeeze()

        group_ref_values[group_idx] = ref_value
        sample_ref_values[group_indices] = ref_value

    target_value = gmean(np.stack(group_ref_values), axis=0)
    norm_factors = target_value / sample_ref_values
    data = data * norm_factors

    if layer is None:
        adata.X = data
    else:
        adata.layers[layer] = data

    return adata if copy else None
