"""Summary metrics that operate on observations or features"""

import warnings

import anndata as ad
import numpy as np

from alphapepttools._matrix import get_matrix
from alphapepttools.pp.transform import detect_special_values


def _resolve_axis(axis: str | int) -> str:
    """Normalize an axis specifier to the canonical "obs" / "var" string.

    Accepts ``"obs"`` or ``0`` for observations (rows), ``"var"`` or ``1`` for features (columns).

    Parameters
    ----------
    axis
        Axis specifier.

    Returns
    -------
    Either ``"obs"`` or ``"var"``.

    Raises
    ------
    ValueError
        If ``axis`` is not one of ``{"obs", "var", 0, 1}``.
    """
    if axis not in ("obs", "var", 0, 1):
        raise ValueError(f"axis must be 'obs', 'var', 0, or 1, got '{axis}'")
    return "obs" if axis in ("obs", 0) else "var"


def total_intensity(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
    axis: str | int = "obs",
    features: list[str] | None = None,
    column: str = "total_intensity",
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
        - "obs" or 0 (default): Calculate total intensity per observation.
          Result is added to adata.obs if `inplace=True`.
        - "var" or 1: Calculate total intensity per feature.
          Result is added to adata.var if `inplace=True`.
    features
        Optional list of specific features (var_names) to include in the sum.
        If None (default), all features are used. Valid only when `axis="obs"`.
    column
        Name of the column to add. Default is "total_intensity".
    inplace
        If True (default), modifies adata inplace and adds the result to adata.obs or adata.var.
        If False, returns the calculated values without modifying adata.

    Returns
    -------
    np.ndarray | None
        If inplace is False, returns the sum values as an array.
        If inplace is True, modifies adata inplace and returns None.
    """
    axis = _resolve_axis(axis)

    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}")

    data = get_matrix(adata, layer)

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
    axis: str | int = "obs",
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
        - "obs" or 0 (default): Count detected features per observation.
          Result is added to adata.obs.
        - "var" or 1: Count detected observations per feature.
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
    axis = _resolve_axis(axis)

    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}")

    data = get_matrix(adata, layer)
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
    axis: str | int = "obs",
    column: str = "fraction_complete",
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
        - "obs" or 0 (default): Calculate fraction of detected features per observation.
          Result is added to adata.obs.
        - "var" or 1: Calculate fraction of detected observations per feature.
          Result is added to adata.var.
    column
        Name of the column to add. Default is "fraction_complete".
    inplace
        If True (default), modifies adata inplace and adds the result to adata.obs or adata.var.
        If False, returns the calculated values without modifying adata.

    Returns
    -------
    np.ndarray | None
        If inplace is False, returns the fraction values as an array.
        If inplace is True, modifies adata inplace and returns None.
    """
    axis = _resolve_axis(axis)

    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}")

    data = get_matrix(adata, layer)
    special_values_mask = detect_special_values(data, verbosity=0)

    # Fraction of detected features per feature or observation
    sum_axis = 1 if axis == "obs" else 0
    result = np.mean(~special_values_mask, axis=sum_axis)

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
