"""Internal helpers for accessing AnnData matrix storage.

anndata types ``.X`` and ``.layers[...]`` as a union of every supported storage
backend (``ndarray | csr_matrix | AwkArray | ZarrArray | CSRDataset | None | ...``).
The pipeline only ever handles dense numpy arrays, so the union is narrowed in
one place here rather than at every call site.
"""

from typing import cast

import anndata as ad
import numpy as np


def get_matrix(adata: ad.AnnData, layer: str | None = None) -> np.ndarray:
    """Return ``adata.X`` or a named layer as a dense numpy array.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Layer to read. If ``None``, ``adata.X`` is returned.

    Returns
    -------
    np.ndarray
        The requested matrix.
    """
    data = adata.X if layer is None else adata.layers[layer]
    return cast("np.ndarray", data)
