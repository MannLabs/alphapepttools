from typing import Literal

import anndata as ad
import numpy as np

STRATEGIES = ["total_mean", "total_median"]


def _validate_strategies(strategy: str) -> None:
    """Verify that valid strategy was selected"""
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

    Example
    -------

    .. code-block:: python

        # Each sample has the same total intensity
        arr = np.array([[1, 1], [2, 0], [0, 2]])
        assert (_total_mean_normalization(arr) == arr).all()

        # Sample 0 has a lower total intensity
        arr = np.array([[0.8, 1], [2, 0], [0, 2]])
        arr_norm = _total_mean_normalization(arr)
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

    See Also
    --------
    alphatools.pp.norm._total_mean_normalization
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
    key_added: str | None = None,
    *,
    copy: bool = False,
) -> ad.AnnData:
    """Normalize measured counts per sample

    Parameters
    ----------
    adata
        Count data
    from_layer:
        Layer that will be normalized. If `None` uses `anndata.AnnData.X`
    to_layer:
        Layer to which the normalized data will be added. If `None` overwrites `anndata.AnnData.X`
    strategy
        Normalization strategy

            - *total_mean* The intensity of each feature is adjusted by a normalizing factor so that the
            total sample intensity is equal to the mean of the total sample intensities across all samples
            - *total_median* The intensity of each feature is adjusted by a normalizing factor so that the
            total sample intensity is equal to the median of the total sample intensities across all samples

    key_added
        If not None, adds normalization factors to column in `adata.obs`

    Returns
    -------
    None | anndata.AnnData
        AnnData object with normalized measurement layer.
        If `copy=False` modifies the anndata object at layer inplace and returns None. If `copy=True`,
        returns a modified copy.

    Example
    -------

    .. code-block:: python

        adata = ad.AnnData(X=np.array([[0.8, 1.0], [2.0, 0.0], [0.0, 2.0]]))
        adata.X
        > np.array([
            [0.8, 1. ],
            [2. , 0. ],
            [0. , 2. ]
        ])

    The anndata object gets normalized in place. Per default, the `.X` attribute will be modified

    .. code-block:: python

        normalize(adata)
        adata.X
        > np.array([
        [0.85925926, 1.07407407],
        [1.93333333, 0.        ],
        [0.        , 1.93333333]]
        )

    Alternatively, we can normalize a different layer

    .. code-block:: python

        adata.layers["normalized"] = adata.X.copy()
        normalize(adata, strategy="total_mean", layer="normalized")
        adata.X
        # Unchanged
        > array([
            [0.8, 1. ],
            [2. , 0. ],
            [0. , 2. ]
        ])

        # Normalized
        adata.layers["normalized"]
        > np.array([
        [0.85925926, 1.07407407],
        [1.93333333, 0.        ],
        [0.        , 1.93333333]]
        )

    Or we return a copy of the object

    .. code-block:: python

        new_adata = normalize(adata, copy=True)
    """
    _validate_strategies(strategy=strategy)

    adata = adata.copy() if copy else adata

    data = adata.layers[layer] if layer is not None else adata.X

    if strategy == "total_mean":
        normalized_data, norm_factors = _total_mean_normalization(data)
    elif strategy == "total_median":
        normalized_data, norm_factors = _total_median_normalization(data)

    # Reassign to anndata
    if layer is None:
        adata.X = normalized_data
    else:
        adata.layers[layer] = normalized_data

    if key_added is not None:
        adata.obs[key_added] = norm_factors

    return adata if copy else None
