"""Pooled median absolute deviation"""

from collections.abc import Callable

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from ._feature_level import _cv

METRICS_KEY = "metrics"
PMAD_KEY = "pmad"
PCV_KEY = "pcv"


def _set_nested_dict(
    dictionary: dict[str, object],
    keys: list[str],
    value: object,
) -> dict[str, object]:
    """Set value in nested dictionary, creating missing keys as needed

    Parameters
    ----------
    dictionary
        Dictionary
    keys
        Path to value, assigned in the order of the indices
    value
        Assigned value at end of path

    Example
    -------

    .. code-block:: python

        _set_nested_dict_keys(dictionary={}, keys=["key1", "key2", "key3"], value="value")
        > {"key1": {"key2": {"key3": value}}}
    """
    current = dictionary

    # Navigate to the parent of the final key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set the final value
    current[keys[-1]] = value
    return dictionary


def _compute_groupwise_metric(
    adata: ad.AnnData, func: Callable[[np.ndarray], float], group_key: str, layer: str | None = None, **kwargs
) -> dict[str, list[float]]:
    """Wrapper function to compute groupwise metrics on data in anndata object

    Extracts data from anndata object (.X/.layers) and computes a groupwise metric

    Parameters
    ----------
    adata
        AnnData object
    func
        Function that computes aggregated metric on data
            - Should expect a numpy array of shape (observations, features) as single argument
            - Should return a single aggregation metric as float
    group_key
        Column in `adata.obs`
    layer
        Layer in adata. If `None`, uses `adata.X`
    **kwargs
        Keyword arguments passed to func

    Returns
    -------
    dict[str, float]
        Per group the computed aggregated metric
    """
    groups = adata.obs.groupby(group_key)
    data = adata.X if layer is None else adata.layers[layer]

    metric = {}
    for group_name, indices in groups.indices.items():
        metric = func(data[indices, :], **kwargs)

        if not isinstance(metric, float):
            raise TypeError(f"`func` needs to return a float value, but returned {type(metric)}")

        metric[group_name] = metric

    return metric


def _pmad(x: np.ndarray) -> float:
    r"""Compute pooled median absolute deviation for a single homogenous group

    .. math ::
        \text{PMAD} = \frac{\sum_{f\in F}{\text{MAD}_{g}(f)}}{|F|}

    - g: Group (here homogenous)
    - f: Feature out of all features
    - MAD: Median absolute deviation
    - |F|: Size of all features

    Parameters
    ----------
    x
        Count data of shape (observations, features)

    Returns
    -------
    float
        Pooled median absolute deviation over features
    """
    # Compute feature-wise MAD (axis=0) and aggregate over all features
    mad = median_abs_deviation(x, axis=0)
    return np.mean(mad)


def _pcv(x: np.ndarray, min_valid: int) -> float:
    r"""Compute pooled coefficient of variation for a single homogenous group

    .. math ::
        \text{PCV} = \frac{\sum_{f\in F}{\text{CV}_{g}(f)}}{|F|}

    - g: Group (here homogenous)
    - f: Feature out of all features
    - CV: Coefficient of variation
    - |F|: Size of all features

    Parameters
    ----------
    x
        Count data of shape (observations, features)

    Returns
    -------
    float
        Pooled coefficient of variation over features while ignoring nans.
    """
    # Compute feature-wise CV (axis=0) and aggregate over all features
    cv = _cv(x, min_valid=min_valid, axis=0)
    return np.nanmean(cv)


def pooled_coefficient_of_variation(
    adata: ad.AnnData, group_key: str, *, min_valid: int = 3, layer: str | None = None, inplace: bool = True
) -> None | pd.DataFrame:
    r"""Compute pooled coefficient of variation within sample groups.

    The pooled coefficient of variation quantifies the variability of features across samples within technical defined groups.
    It is particularly useful for assessing the variability of a method in technical
    replicates across batches.

    For each group :math:`g` in a set of sample groups :math:`G`, the  is calculated as the average
    coefficient of variation across all features :math:`f \in F`:

    .. math::

        \text{PCV}_g = \frac{1}{|F|} \sum_{f \in F} \mathrm{CV}_g(f)

    where :math:`\mathrm{CV}_g(f)` is the coefficient of variation of feature :math:`f` within group :math:`g`.

    In the original publication, the PCV was computed for each group and used to compare different normalization strategies.
    Lower PCV values with technical replicates indicate reduced intra-group variability and may suggest improved normalization.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    group_key
        Column in `adata.obs` that defines the sample groups to evaluate (e.g., biological replicates or batches).
    min_valid
        Minimal number of valid samples to
    layer
        Layer for which the metric is computed.
    inplace
        If `True`, the results are added to `adata.uns['pmad']`. The object is changed in place
        If `False`, a :class:`pandas.DataFrame` with the PMAD values is returned.

    Returns
    -------
    AnnData or pandas.DataFrame
        If `inplace=True`, modifies the input `adata` with PCV values stored in `adata.uns["metrics"]["pcv"]`.
        If `inplace=False`, returns a DataFrame containing PCV values per group.

    See Also
    --------
    :func:`alphapepttools.metrics.coefficient_of_variation`

    References
    ----------
    - Chawade, A., Alexandersson, E. & Levander, F. Normalyzer: A Tool for Rapid Evaluation of Normalization Methods
    for Omics Data Sets. J. Proteome Res. 13, 3114-3120 (2014).

    """
    pcv_groupwise = _compute_groupwise_metric(
        adata=adata, func=_pcv, group_key=group_key, layer=layer, min_valid=min_valid
    )

    if inplace:
        adata.uns = _set_nested_dict(adata.uns, value=pcv_groupwise, keys=[METRICS_KEY, PCV_KEY])
    return pd.DataFrame.from_dict(pcv_groupwise, orient="index", columns=[PCV_KEY])


def pooled_median_absolute_deviation(
    adata: ad.AnnData, group_key: str, *, layer: str | None = None, inplace: bool = True
) -> None | pd.DataFrame:
    r"""Compute pooled median absolute deviation (PMAD) within sample groups.

    The PMAD quantifies the variability of features across samples within biologically defined groups.
    It is particularly useful for assessing the performance of normalization approaches, especially when groups
    are expected to be biologically homogeneous (Arend et al., 2025).

    For each group :math:`g` in a set of sample groups :math:`G`, the PMAD is calculated as the average
    median absolute deviation (MAD) across all features :math:`f \in F`:

    .. math::

        \text{PMAD}_g = \frac{1}{|F|} \sum_{f \in F} \mathrm{MAD}_g(f)

    where :math:`\mathrm{MAD}_g(f)` is the median absolute deviation of feature :math:`f` within group :math:`g`.

    In the original publication, PMAD was computed for each group and used to compare different normalization strategies.
    Lower PMAD values indicate reduced intra-group variability and may suggest improved normalization.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    group_key
        Column in `adata.obs` that defines the sample groups to evaluate (e.g., biological replicates or batches).
    layer
        Layer for which the metric is computed.
    inplace
        If `True`, the results are added to `adata.uns['pmad']`. The object is changed in place
        If `False`, a :class:`pandas.DataFrame` with the PMAD values is returned.

    Returns
    -------
    AnnData or pandas.DataFrame
        If `inplace=True`, modifies the input `adata` with PMAD values stored in `adata.uns["metrics"]["pmad"]`.
        If `inplace=False`, returns a DataFrame containing PMAD values per group.

    Notes
    -----
    Some normalization approaches, such as MAD-based normalization, explicitly minimize intra-group variability.
    In such cases, PMAD may directly reflect the objective of the normalization procedure, and its interpretation
    should be made alongside other complementary metrics.

    References
    ----------
    - Arend, L. et al. Systematic evaluation of normalization approaches in tandem mass tag and label-free protein
      quantification data using PRONE. *Briefings in Bioinformatics*, 26, bbaf201 (2025).
    """
    pmad_groupwise = _compute_groupwise_metric(adata=adata, func=_pmad, group_key=group_key, layer=layer)

    if inplace:
        adata.uns = _set_nested_dict(adata.uns, value=pmad_groupwise, keys=[METRICS_KEY, PMAD_KEY])
    return pd.DataFrame.from_dict(pmad_groupwise, orient="index", columns=[PMAD_KEY])
