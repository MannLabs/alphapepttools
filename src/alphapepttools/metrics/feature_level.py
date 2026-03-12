"""Feature level metrics"""

import warnings

import anndata as ad
import numpy as np

from alphapepttools.pp.transform import detect_special_values


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
    std = np.nanstd(data, axis=axis)
    mean = np.nanmean(data, axis=axis)
    cv = std / np.where(mean == 0, np.nan, mean)
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
        AnnData object with computed CVs added to `adata.var[key_added]`.
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


def total_intensity(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
    axis: str = "obs",
    features: list[str] | None = None,
    column: str | None = None,
    inplace: bool = True,
) -> np.ndarray | None:
    """Calculate sum of intensity per observation or per feature.

    Parameters
    ----------
    adata
        AnnData object
    layer
        Name of the layer to compute the sum on. If None (default), the data matrix X is used.
    axis
        Axis along which to calculate the sum.
        - "obs" (default): Calculate total intensity per observation.
          Result is added to adata.obs if `inplace=True`.
        - "var": Calculate total intensity per feature.
          Result is added to adata.var if `inplace=True`.
    features
        Optional list of specific features (var_names) to include in the sum.
        If None (default), all features are used. Valid only when `axis="obs"`.
    column
        Name of the column to add. If None, defaults to "total_intensity".
    inplace
        If True (default), modifies adata inplace and adds the result to adata.obs or adata.var.
        If False, returns the calculated values without modifying adata.

    Returns
    -------
    np.ndarray | None
        If inplace is False, returns the sum values as an array.
        If inplace is True, modifies adata inplace and returns None.
    """
    if axis not in ("obs", "var"):
        raise ValueError(f"axis must be 'obs' or 'var', got '{axis}'")

    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}")

    if column is None:
        column = "total_intensity"

    data = adata.X if layer is None else adata.layers[layer]

    if features is not None and axis == "obs":
        missing = set(features) - set(adata.var_names)
        if missing == set(features):
            raise ValueError("None of the specified features were found in adata.var_names")
        if missing:
            warnings.warn(f"The following features were not found in adata.var_names: {missing}")
        feature_mask = adata.var_names.isin(features)
        data = data[:, feature_mask]

    result = np.nansum(data, axis=1) if axis == "obs" else np.nansum(data, axis=0)

    if inplace:
        if axis == "obs":
            adata.obs[column] = result
        else:
            adata.var[column] = result
        return None
    return result


def number_detected(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
    axis: str = "obs",
    column: str = "number_detected",
    inplace: bool = True,
) -> np.ndarray | None:
    """Count the number of detected features per observation or detected observations per feature.

    A value is considered detected if it is not a special value
    (NaN, zero, negative, or infinite).

    Parameters
    ----------
    adata
        AnnData object
    layer
        Name of the layer to use. If None (default), the data matrix X is used.
    axis
        Axis along which to calculate the count.
        - "obs" (default): Count detected features per observation.
          Result is added to adata.obs.
        - "var": Count detected observations per feature.
          Result is added to adata.var.
    column
        Name of the column to add. Default is "number_detected".
    inplace
        If True (default), modifies adata inplace and adds the result to adata.obs or adata.var.
        If False, returns the calculated values without modifying adata.

    Returns
    -------
    np.ndarray | None
        If inplace is False, returns the count values as an array.
        If inplace is True, modifies adata inplace and returns None.
    """
    if axis not in ("obs", "var"):
        raise ValueError(f"axis must be 'obs' or 'var', got '{axis}'")

    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}")

    data = adata.X if layer is None else adata.layers[layer]
    special_values_mask = detect_special_values(data, verbosity=0)

    result = np.sum(~special_values_mask, axis=1) if axis == "obs" else np.sum(~special_values_mask, axis=0)

    if inplace:
        if axis == "obs":
            adata.obs[column] = result
        else:
            adata.var[column] = result
        return None
    return result


def fraction_complete(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
    axis: str = "obs",
    column: str | None = None,
    inplace: bool = True,
) -> np.ndarray | None:
    """Calculate the fraction of detected values per observation or per feature.

    A value is considered detected if it is not a special value
    (NaN, zero, negative, or infinite).

    Parameters
    ----------
    adata
        AnnData object
    layer
        Name of the layer to use. If None (default), the data matrix X is used.
    axis
        Axis along which to calculate the fraction.
        - "obs" (default): Calculate fraction of detected features per observation.
          Result is added to adata.obs.
        - "var": Calculate fraction of detected observations per feature.
          Result is added to adata.var.
    column
        Name of the column to add. If None, defaults to "fraction_complete".
    inplace
        If True (default), modifies adata inplace and adds the result to adata.obs or adata.var.
        If False, returns the calculated values without modifying adata.

    Returns
    -------
    np.ndarray | None
        If inplace is False, returns the fraction values as an array.
        If inplace is True, modifies adata inplace and returns None.
    """
    if axis not in ("obs", "var"):
        raise ValueError(f"axis must be 'obs' or 'var', got '{axis}'")

    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}")

    if column is None:
        column = "fraction_complete"

    data = adata.X if layer is None else adata.layers[layer]
    special_values_mask = detect_special_values(data, verbosity=0)

    if axis == "obs":
        # Per observation: count detected features / total features
        n_detected = np.sum(~special_values_mask, axis=1)
        n_total = adata.shape[1]
    else:  # axis == "var"
        # Per feature: count detected observations / total observations
        n_detected = np.sum(~special_values_mask, axis=0)
        n_total = adata.shape[0]

    result = n_detected / n_total

    if inplace:
        if axis == "obs":
            adata.obs[column] = result
        else:
            adata.var[column] = result
        return None
    return result


def calculate_qc_metrics(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
) -> None:
    """Calculate all QC metrics and add them to adata.obs.

    This function computes and adds the following metrics:
    - total_intensity: Sum of intensities per observation and per feature
    - num_features_detected: Number of detected features per observation and per feature
    - fraction_complete: Fraction of detected features per observation and per feature

    Parameters
    ----------
    adata
        AnnData object
    layer
        Name of the layer to use. If None (default), the data matrix X is used.

    Returns
    -------
    None
        Modifies adata inplace by adding the following columns to adata.obs:
        - ``adata.obs["total_sample_intensity"]``: Sum of intensities per observation
        - ``adata.obs["num_features_detected"]``: Number of detected features per observation
        - ``adata.obs["fraction_detected_features"]``: Fraction of detected features per observation
        - ``adata.var["total_feature_intensity"]``: Sum of intensities per feature across all observations
        - ``adata.var["num_samples_detected"]``: Number of samples in which the features was detected
        - ``adata.var["fraction_detected_samples"]``: Fraction of detected observations per feature (across all observations)

    """
    total_intensity(adata, layer=layer, axis="obs", column="total_sample_intensity")
    number_detected(adata, layer=layer, axis="obs", column="num_features_detected")
    fraction_complete(adata, layer=layer, axis="obs", column="fraction_detected_features")

    total_intensity(adata, layer=layer, axis="var", column="total_feature_intensity")
    number_detected(adata, layer=layer, axis="var", column="num_samples_detected")
    fraction_complete(adata, layer=layer, axis="var", column="fraction_detected_samples")
