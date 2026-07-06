"""Internal helpers for accessing AnnData storage.

anndata types ``.X``/``.layers[...]`` as a union of every supported storage backend
(``ndarray | csr_matrix | AwkArray | ZarrArray | CSRDataset | None | ...``) and
``.obs``/``.var`` as ``DataFrame | Dataset2D`` (with column access yielding
``Series | XDataArray``). The pipeline only ever handles dense numpy arrays and
pandas frames, so the unions are narrowed in one place here rather than at every
call site.
"""

from typing import cast

import anndata as ad
import numpy as np
import pandas as pd


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


def get_obs(adata: ad.AnnData) -> pd.DataFrame:
    """Return ``adata.obs`` as a pandas DataFrame.

    Parameters
    ----------
    adata
        Annotated data matrix.

    Returns
    -------
    pd.DataFrame
        The observation annotations.
    """
    return cast("pd.DataFrame", adata.obs)


def get_var(adata: ad.AnnData) -> pd.DataFrame:
    """Return ``adata.var`` as a pandas DataFrame.

    Parameters
    ----------
    adata
        Annotated data matrix.

    Returns
    -------
    pd.DataFrame
        The variable annotations.
    """
    return cast("pd.DataFrame", adata.var)
