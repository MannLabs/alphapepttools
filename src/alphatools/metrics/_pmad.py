"""Pooled median absolute deviation"""

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

METRICS_KEY = "metrics"
PMAD_KEY = "pmad"


def _set_recursive_dict_keys(dictionary: dict[str, Any], value: Any, keys: list[str]) -> dict[str, Any]:  # noqa: ANN401
    """Set the value in a nested dictionary and create non-existent keys"""
    k = keys.pop(0)

    if len(keys) == 0:
        dictionary[k] = value
        return dictionary

    if k not in dictionary:
        dictionary[k] = _set_recursive_dict_keys({}, value=value, keys=keys)
    else:
        dictionary[k] = _set_recursive_dict_keys(dictionary[k], value=value, keys=keys)

    return dictionary


def _pmad(adata: ad.AnnData) -> float:
    r"""Compute pooled median absolute deviation for a single homogenous group

    .. math ::
        \text{PMAD} = \frac{\sum_{f\in F}{\text{MAD_g(f)}}}{|F|}

    - g: Group (here homogenous)
    - f: Feature out of all features
    - MAD: Median absolute deviation
    - |F|: Size of all features
    """
    # Compute feature-wise MAD (axis=0) and aggregate over all features
    mad = median_abs_deviation(adata.X, axis=0)
    return np.mean(mad)


def pmad(adata: ad.AnnData, group_key: str, *, inplace: bool = True) -> ad.AnnData | pd.DataFrame:
    r"""Pooled median absolute deviation

    Quantifies the variation of counts between samples of a defined group of samples.
    If the grouping is expected to represent a biologically homogenous collection, the PMAD can
    be used to compare the effect of normalization approaches, as proposed in (Arend et al, 2025).

    The pooled median absolute deviation within a group of samples g out of all sample groups G is defined as
    the median absolute deviation over all |F| features $f \in F$
    of the group

    .. math ::
        \text{PMAD_g} = \frac{\sum_{g, f\in F}{\text{MAD_g(f)}}}{|F|}

    In the original publication, the PMAD is computed for every sample group and compared between
    normalization approaches. Lower PMADs indicate lower intra-group variability which might indicate
    a better normalization.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` object
    group_key
        Grouping variable. Column in `adata.obs` representing a meaningful biological group

    Notes
    -----
    Note that normalization approaches such as the Median absolute deviation normalization explicitly normalize
    by the MAD, i.e. they explicitly optimize for this metric. The PMAD should therefore be only considered in
    conjunction with other metrics.

    References
    ----------
    - Arend, L. et al. Systematic evaluation of normalization approaches in tandem mass tag and label-free protein quantification data using PRONE. Briefings in Bioinformatics 26, bbaf201 (2025).
    """
    groups = adata.obs.groupby(group_key)

    pmad_groupwise = {}
    for group_name, indices in groups.indices.items():
        pmad_groupwise[group_name] = _pmad(adata[indices, :])

    if inplace:
        adata.uns = _set_recursive_dict_keys(adata.uns, value=pmad_groupwise, keys=[METRICS_KEY, PMAD_KEY])
        return adata
    return pd.DataFrame.from_dict(pmad_groupwise, orient="index", columns=[PMAD_KEY])
