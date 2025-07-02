"""Pooled median absolute deviation"""

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

METRICS_KEY = "metrics"
PMAD_KEY = "pmad"


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


def pooled_median_absolute_deviation(
    adata: ad.AnnData, group_key: str, *, inplace: bool = True
) -> ad.AnnData | pd.DataFrame:
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
    group_key : str
        Column in `adata.obs` that defines the sample groups to evaluate (e.g., biological replicates or batches).
    inplace : bool, default: True
        If `True`, the results are added to `adata.uns['pmad']`.
        If `False`, a :class:`pandas.DataFrame` with the PMAD values is returned.

    Returns
    -------
    AnnData or pandas.DataFrame
        If `inplace=True`, returns the input `adata` with PMAD values stored in `adata.uns["metrics"]["pmad"]`.
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
    groups = adata.obs.groupby(group_key)

    pmad_groupwise = {}
    for group_name, indices in groups.indices.items():
        pmad_groupwise[group_name] = _pmad(adata[indices, :].X)

    if inplace:
        adata.uns = _set_nested_dict(adata.uns, value=pmad_groupwise, keys=[METRICS_KEY, PMAD_KEY])
        return adata
    return pd.DataFrame.from_dict(pmad_groupwise, orient="index", columns=[PMAD_KEY])
