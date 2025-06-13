import warnings
from typing import Literal

import anndata as ad
import numpy as np

STRATEGIES = ["mean"]


def _validate_strategies(strategy: str) -> None:
    """Verify that valid strategy was selected"""
    if strategy not in STRATEGIES:
        raise ValueError(f"`strategy` must be one of {STRATEGIES}, not {strategy}")


def _mean_normalization(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean normalization

    Normalizes total intensity in each sample (row) to mean of the total intensities

    Parameters
    ----------
    data
        Count data of shape (samples, features)

    Example
    -------

    .. code-block:: python

        # Each sample is the same
        arr = np.array([[1, 1], [2, 0], [0, 2]])
        assert (_mean_normalization(arr) == arr).all()

        # Sample 0 has a lower total intensity
        arr = np.array([[0.8, 1], [2, 0], [0, 2]])
        arr_norm = _mean_normalization(arr)
        arr_norm.sum(axis=1)
        > array([1.93333333, 1.93333333, 1.93333333])
    """
    # Compute sample-wise means
    total_counts = data.sum(axis=1)
    norm_factor = total_counts.mean() / total_counts

    return data * norm_factor.reshape(-1, 1), norm_factor


def normalize(
    adata: ad.AnnData,
    from_layer: str | None = None,
    to_layer: str | None = None,
    strategy: Literal["mean"] = "mean",
    key_added: str | None = None,
) -> ad.AnnData:
    """Normalize measured counts per sample

    Parameters
    ----------
    adata
        Count data
    layer:
        Count layer. If `None` uses `anndata.AnnData.X`
    strategy
        Normalization strategy

            - *mean* The intensity of each feature is adjusted by a normalizing factor so that the
            total sample intensity is equal to the mean of the total sample intensities across all samples
    key_added
        If not None, adds normalization factors to column in `adata.obs`
    """
    _validate_strategies(strategy=strategy)

    if (not isinstance(key_added, str)) and (key_added is not None):
        raise TypeError(f"`key_added` must be str not {type(key_added)}")

    data = adata.layers[from_layer] if from_layer is not None else adata.X

    if strategy == "mean":
        normalized_data, norm_factors = _mean_normalization(data)

    # Reassign to anndata
    if to_layer is None:
        adata.X = normalized_data
    else:
        if to_layer in adata.layers:
            warnings.warn(f"Layer {to_layer} already in adata. Overwriting in memory.", stacklevel=2)
        adata.layers[to_layer] = normalized_data

    if key_added is not None:
        adata.obs[key_added] = norm_factors
