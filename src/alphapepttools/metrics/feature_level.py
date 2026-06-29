"""Feature level metrics"""

import numbers
import warnings
from collections.abc import Callable, Hashable, MutableMapping

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from alphapepttools._matrix import get_matrix

METRICS_KEY = "metrics"
PMAD_KEY = "pmad"
PCV_KEY = "pcv"


def _cv(data: np.ndarray, *, min_valid: int = 3, axis: int = 0) -> np.ndarray:
    """Compute the coefficient of variation

    Parameters
    ----------
    data
        Array of shape (observations, features)
    min_valid
        Minimum number of samples with non-na values to compute the value
    axis
        Axis along which to compute CV (defaults to feature-wise)

    Returns
    -------
    1D Array with length of axis with computed CVs and `nan` where the
    number of non-na values is smaller than min_valid

    """
    # Silence runtime warnings from nanmean/nanstd. The NaNs are anyway masked out by `min_valid` below so the warnings carry no information for the caller.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0")
        std = np.nanstd(data, axis=axis)
        mean = np.nanmean(data, axis=axis)
    cv = std / np.where(mean == 0, np.nan, mean)
    valid_count = np.sum(~np.isnan(data), axis=axis)
    cv[valid_count < min_valid] = np.nan
    return cv


def coefficient_of_variation(
    adata: ad.AnnData,
    *,
    group_column: str | None = None,
    min_valid: int = 3,
    key_added: str = "cv",
    layer: str | None = None,
    copy: bool = False,
) -> None | ad.AnnData:
    r"""Coefficient of variation

    Compute the coefficient of variation (CV) per feature, either across all samples or
    within sample groups.

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
    group_column
        Column name in `adata.obs` defining groups for groupwise CV computation.
        If `None` (default), one CV per feature is computed across all samples and stored
        as a column in `adata.var[key_added]`.
        If specified, one CV per feature is computed within each group separately and stored
        as a DataFrame in `adata.varm[key_added]` with `adata.var_names` as the index and
        group names as columns.
    min_valid
        Minimum number of samples required to estimate the CV. Will be set to `NaN` otherwise
    key_added
        Name of the column added to `adata.var` (ungrouped) or the key added to `adata.varm` (grouped)
    layer
        Name of the layer to compute metric on. If None (default), the data matrix X is used
    copy
        If `False` (default), modifies `adata` inplace and returns `None`. If `True`, returns a copy of the `adata` object.

    Returns
    -------
    AnnData object with computed CVs.
    If `group_column=None`, results are written to `adata.var[key_added]`.
    If `group_column` is set, results are written to `adata.varm[key_added]` as a DataFrame.
    If `copy=False` modifies the anndata object inplace and returns None. If `copy=True`,
    returns a modified copy

    Raises
    ------
    ValueError
        If `group_column` is set and contains NaN values

    Examples
    --------
    Compute one CV per feature across all samples:

    .. code-block:: python

        import numpy as np
        import anndata as ad
        import pandas as pd
        import alphapepttools as apt

        adata = ad.AnnData(
            X=np.array([[1, 2], [5, 1], [6, 6], [9, 3], [4, 8], [7, 4]]),
            obs=pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"]}),
            var=pd.DataFrame(index=["protein1", "protein2"]),
        )

        apt.metrics.coefficient_of_variation(adata)
        print(adata.var["cv"])  # one CV per feature across all samples

    Compute one CV per feature per group, e.g. for replicate groups:

    .. code-block:: python

        apt.metrics.coefficient_of_variation(adata, group_column="group")
        print(adata.varm["cv"])
        # DataFrame indexed by var_names, columns are group labels:
        #            A    B
        # protein1  0.540062  0.308221
        # protein2  0.720082  0.432049

    Notes
    -----
    The CV only considers non-missing values and should be computed before imputation.
    Features with fewer than `min_valid` non-missing values will return NaN for CV

    """
    adata = adata.copy() if copy else adata
    data = get_matrix(adata, layer)

    if group_column is None:
        adata.var[key_added] = _cv(data, min_valid=min_valid, axis=0)
    else:
        if pd.isna(adata.obs[group_column]).any():
            raise ValueError(
                f"`group_column` '{group_column}' contains NaNs. "
                "Drop or relabel the corresponding observations before computing groupwise CVs."
            )

        groups = adata.obs.groupby(group_column, dropna=True, observed=False).indices
        adata.varm[key_added] = pd.DataFrame(
            {name: _cv(data[idx, :], min_valid=min_valid, axis=0) for name, idx in groups.items()},
            index=adata.var_names,
        )

    return adata if copy else None


def _set_nested_dict(
    dictionary: MutableMapping[str, object],
    keys: list[str],
    value: object,
) -> MutableMapping[str, object]:
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


def _compute_pooled_groupwise_metric(
    adata: ad.AnnData, func: Callable[..., float], group_column: str, layer: str | None = None, **kwargs
) -> dict[Hashable, float]:
    """Wrapper function to compute pooled groupwise metrics on data in anndata object

    Extracts data from anndata object (.X/.layers) and computes a groupwise metric

    Parameters
    ----------
    adata
        AnnData object
    func
        Function that computes aggregated metric on data
            - Should expect a numpy array of shape (observations, features) as single argument
            - Should return a single aggregation metric as numeric value (float, int)
    group_column
        Column in `adata.obs`
    layer
        Layer in adata. If `None`, uses `adata.X`
    **kwargs
        Keyword arguments passed to func

    Returns
    -------
    dict[str, float]
        Per group the computed aggregated metric

    Examples
    --------
    .. code-block:: python

        from alphapepttools.metrics.feature_level import _compute_pooled_groupwise_metric, _pcv

        # Compute pooled CV for each experimental condition
        pcv_groupwise = _compute_pooled_groupwise_metric(adata=adata, func=_pcv, group_column="condition", min_valid=3)
        print(pcv_groupwise)  # {'control': 0.25, 'treated': 0.31}

    """
    groups = adata.obs.groupby(group_column, observed=False)
    data = get_matrix(adata, layer)

    metrics = {}
    for group_name, indices in groups.indices.items():
        result = func(data[indices, :], **kwargs)

        # Check if result is a numeric scalar (handles numpy types and python built-ins)
        if not isinstance(result, numbers.Real):
            raise TypeError(f"`func` needs to return a numeric value, but returned {type(result).__name__}: {result}")

        metrics[group_name] = result

    return metrics


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
    min_valid
        Minimum number of valid values required for CV computation

    Returns
    -------
    Pooled coefficient of variation over features while ignoring nans

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from alphapepttools.metrics.feature_level import _pcv

        # Data for one group (5 samples, 3 features)
        data = np.array([[10, 20, 30], [11, 22, 31], [9, 19, 28], [12, 21, 32], [8, 20, 30]])
        pcv_value = _pcv(data, min_valid=3)
        print(f"PCV: {pcv_value:.3f}")

    """
    # Compute feature-wise CV (axis=0) and aggregate over all features
    cv = _cv(x, min_valid=min_valid, axis=0)
    return np.nanmean(cv)


def pooled_coefficient_of_variation(
    adata: ad.AnnData,
    group_column: str,
    *,
    min_valid: int = 3,
    layer: str | None = None,
    inplace: bool = True,
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
    adata
        Annotated data matrix
    group_column
        Column in `adata.obs` that defines the sample groups to evaluate (e.g., biological replicates or batches)
    min_valid
        Minimal number of valid samples to compute CV
    layer
        Layer for which the metric is computed
    inplace
        If `True`, the results are added to `adata.uns['metrics']['pcv']`. The object is changed in place.
        If `False`, a DataFrame with the PCV values is returned

    Returns
    -------
    If `inplace=True`, modifies the input `adata` with PCV values stored in `adata.uns["metrics"]["pcv"]`.
    If `inplace=False`, returns a DataFrame containing PCV values per group

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import anndata as ad
        import pandas as pd
        import alphapepttools as apt

        # Create example data
        adata = ad.AnnData(
            X=np.array([[1, 2], [5, 1], [6, 6], [9, 3], [4, 8], [7, 4]]),
            obs=pd.DataFrame({"replicate_group": ["A", "A", "A", "B", "B", "B"]}),
            var=pd.DataFrame(index=["feature1", "feature2"]),
        )

        # Compute PCV for technical replicates
        apt.metrics.pooled_coefficient_of_variation(adata, group_column="replicate_group")

        # Access results
        print(adata.uns["metrics"]["pcv"])
        # # {'A': 0.63, 'B': 0.37}

    See Also
    --------
    :func:`alphapepttools.metrics.coefficient_of_variation`

    References
    ----------
    - Chawade, A., Alexandersson, E. & Levander, F. Normalyzer: A Tool for Rapid Evaluation of Normalization Methods
    for Omics Data Sets. J. Proteome Res. 13, 3114-3120 (2014).

    """
    pcv_groupwise = _compute_pooled_groupwise_metric(
        adata=adata, func=_pcv, group_column=group_column, layer=layer, min_valid=min_valid
    )

    if inplace:
        adata.uns = _set_nested_dict(adata.uns, value=pcv_groupwise, keys=[METRICS_KEY, PCV_KEY])
    return pd.DataFrame.from_dict(pcv_groupwise, orient="index", columns=pd.Index([PCV_KEY]))


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
    Pooled median absolute deviation over features

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from alphapepttools.metrics.feature_level import _pmad

        # Data for one group (5 samples, 3 features)
        data = np.array([[1, 2, 3], [1.1, 2.2, 3.1], [0.9, 1.9, 2.8], [1.2, 2.1, 3.2], [0.8, 2.0, 3.0]])
        pmad_value = _pmad(data)
        print(f"PMAD: {pmad_value:.3f}")

    """
    # Compute feature-wise MAD (axis=0) and aggregate over all features
    mad = median_abs_deviation(x, axis=0)
    return np.mean(mad)


def pooled_median_absolute_deviation(
    adata: ad.AnnData,
    group_column: str,
    *,
    layer: str | None = None,
    inplace: bool = True,
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
    adata
        Annotated data matrix
    group_column
        Column in `adata.obs` that defines the sample groups to evaluate (e.g., biological replicates or batches)
    layer
        Layer for which the metric is computed
    inplace
        If `True`, the results are added to `adata.uns['metrics']['pmad']`. The object is changed in place.
        If `False`, a DataFrame with the PMAD values is returned

    Returns
    -------
    If `inplace=True`, modifies the input `adata` with PMAD values stored in `adata.uns["metrics"]["pmad"]`.
    If `inplace=False`, returns a DataFrame containing PMAD values per group

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import anndata as ad
        import pandas as pd
        import alphapepttools as apt

        # Create example data
        adata = ad.AnnData(
            X=np.array([[1, 2], [5, 1], [6, 6], [9, 3], [4, 8], [7, 4]]),
            obs=pd.DataFrame({"condition": ["ctrl", "ctrl", "ctrl", "treat", "treat", "treat"]}),
            var=pd.DataFrame(index=["feature1", "feature2"]),
        )

        # Compute PMAD for biological conditions
        apt.metrics.pooled_median_absolute_deviation(adata, group_column="condition")

        # Access results
        print(adata.uns["metrics"]["pmad"])
        # {'ctrl': 1, 'treat': 1.5}

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
    pmad_groupwise = _compute_pooled_groupwise_metric(adata=adata, func=_pmad, group_column=group_column, layer=layer)

    if inplace:
        adata.uns = _set_nested_dict(adata.uns, value=pmad_groupwise, keys=[METRICS_KEY, PMAD_KEY])
    return pd.DataFrame.from_dict(pmad_groupwise, orient="index", columns=pd.Index([PMAD_KEY]))
