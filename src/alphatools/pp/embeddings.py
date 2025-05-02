import logging

import anndata as ad
import numpy as np
from scanpy import preprocessing as scpp

# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_core_proteome_mask(data: ad.AnnData, layer: str, namecol: str = "isCore") -> ad.AnnData:
    """Adds a column of booleans to the adata.var table

    indicates whether the protein appears in 100% of the samples (TRUE) or not (FALSE)Handle missing values in the data matrix.
    this is useful for running PCA with the core proteome as "mask_var".

    Parameters
    ----------
    data: ad.AnnData
        AnnData od the protein data. The protein matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to genes.
    layer: str
        Which layer of the AnnData to use for PCA (relevant when imputation is also in place).
    namecol: str, optional (default: "isCore")
        which name to assign to the new column in AnnData.var.

    Returns
    -------
    adata : ad.AnnData
        AnnData object with the added core var meta data.
    """
    logger.info("Adding core proteome mask to feature metadata")

    # checks
    if not isinstance(data, (ad.AnnData)):
        raise TypeError("Data must be either AnnData object")
    if layer and layer != "X" and layer not in data.layers:
        raise ValueError(f"Layer {layer} not found in AnnData object")
    mat = data.X if layer is None else data.layers[layer]

    # indicate which proteins are in the core proteome
    if namecol in data.var.columns:
        logger.warning(f"'{namecol}' exists in data.var, overwriting")

    data.var[namecol] = ~np.isnan(mat).any(axis=0)
    logger.info(f"'{namecol}' column added to data.var to classify core proteins")

    return data


def run_pca(
    data: ad.AnnData,
    n_comps: int | None = None,
    layer: str | None = None,
    feature_meta_data_mask: str | None = None,
    pca_kwargs: dict | None = None,
) -> ad.AnnData | np.ndarray:
    """Principal component analysis :cite:p:`Pedregosa2011`.

    Computes PCA coordinates, loadings and variance decomposition. Uses the
    implementation of Scanpy (v 1.10.4), which in turn uses implementation of
    *scikit-learn* :cite:p:`Pedregosa2011`.

    Parameters
    ----------
    data: ad.AnnData
        The (annotated) data matrix of shape `n_obs` X `n_vars`.
        Rows correspond to cells and columns to genes.
    n_comps: int, optional (default: 50)
        Number of principal components to compute. Defaults to 50, or 1 - minimum
        dimension size of selected representation.
    layer: str, optional (default: "X")
        If provided, which element of layers to use for PCA.
        If a np.array is provided, it is used directly.
    feature_meta_data_mask: str, optional (default: None)
        If provided, the colname in `adata.var` to use as a mask for
        the features to be used in PCA. This is useful for running PCA with the
        core proteome as "mask_var".
        If None, all features are used (data should not iclude NaNs!).
    pca_kwargs: dict, optional
        Additional keyword arguments for the PCA function. By default None.

    Returns
    -------
    (as output from the scanpy.pp.pca function)
    unless changed in the kwargs passed on to scanpy, an updated `AnnData` object.
    Sets the following fields:
    `.obsm['X_pca' | key_added]` : :class:`~scipy.sparse.csr_matrix` | :class:`~scipy.sparse.csc_matrix` | :class:`~numpy.ndarray` (shape `(adata.n_obs, n_comps)`)
        PCA representation of data.
    `.varm['PCs' | key_added]` : :class:`~numpy.ndarray` (shape `(adata.n_vars, n_comps)`)
        The principal components containing the loadings.
    `.uns['pca' | key_added]['variance_ratio']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Ratio of explained variance.
    `.uns['pca' | key_added]['variance']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Explained variance, equivalent to the eigenvalues of the
        covariance matrix.
    """
    logger.info("computing PCA")
    pca_kwargs = pca_kwargs or {}

    if not isinstance(data, (ad.AnnData)):
        raise TypeError("Data must be either AnnData object or numpy array")
    if layer and layer != "X" and layer not in data.layers:
        raise ValueError(f"Layer {layer} not found in AnnData object")

    # Add feature mask to kwargs if provided
    if feature_meta_data_mask is not None:
        if feature_meta_data_mask not in data.var.columns:
            raise ValueError(f"Column {feature_meta_data_mask} not found in data.var")
        pca_kwargs["mask_var"] = data.var[feature_meta_data_mask]

    return scpp.pca(data, n_comps=n_comps, layer=layer, **pca_kwargs)
