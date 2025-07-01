import warnings
from typing import Literal

import anndata as ad
import numpy as np

STRATEGIES = ["total_mean"]


def _validate_strategies(strategy: str) -> None:
    """Verify that valid strategy was selected"""
    if strategy not in STRATEGIES:
        raise ValueError(f"`strategy` must be one of {STRATEGIES}, not {strategy}")


def _total_mean_normalization(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total normalization

    Normalizes total intensity in each sample (row) to mean of the total intensities

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
    total_counts = data.sum(axis=1)
    norm_factor = total_counts.mean() / total_counts

    return data * norm_factor.reshape(-1, 1), norm_factor


def normalize(
    adata: ad.AnnData,
    from_layer: str | None = None,
    to_layer: str | None = None,
    strategy: Literal["total_mean"] = "total_mean",
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

            - *total_mean* The intensity of each feature is adjusted by a normalizing factor so that the
            total sample intensity is equal to the mean of the total sample intensities across all samples
    key_added
        If not None, adds normalization factors to column in `adata.obs`

    Returns
    -------
    None
        Modifies the :class:`anndata.AnnData` object in place

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

    Alternatively, we can generate a new layer

    .. code-block:: python
        normalize(adata, strategy="total_mean", to_layer="normalized")
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
    """
    _validate_strategies(strategy=strategy)

    data = adata.layers[from_layer] if from_layer is not None else adata.X

    if strategy == "total_mean":
        normalized_data, norm_factors = _total_mean_normalization(data)

    # Reassign to anndata
    if to_layer is None:
        adata.X = normalized_data
    else:
        if to_layer in adata.layers:
            warnings.warn(f"Layer {to_layer} already in adata. Overwriting in memory.", stacklevel=2)
        adata.layers[to_layer] = normalized_data

    if key_added is not None:
        adata.obs[key_added] = norm_factors
