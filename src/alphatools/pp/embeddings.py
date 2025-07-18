import logging

import anndata as ad
import numpy as np
import scanpy as sc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_inputs_for_dim_reduction(
    adata: ad.AnnData, layer: str | None, dim_space: str, meta_data_mask_column_name: str | None
) -> None:
    """Check inputs for PCA and other dimensionality reduction methods.

    Parameters
    ----------
    adata: ad.AnnData
        The (annotated) data matrix of shape `n_obs` X `n_vars`.
        Rows correspond to cells and columns to genes.
    layer: str, optional (default: None)
        If provided, which element of layers to use for PCA.
        If None, the `.X` attribute of `adata` is used.
    dim_space: str, optional (default: "obs")
        The dimension to project PCA on. Can be either "obs" (default) for
        sample projection or "var" for feature projection.
    meta_data_mask_column_name: str, optional (default: None)
        If provided, the colname in `adata.var` to use as a mask for
        the features to be used in PCA. must be of boolean dtype.

    Raises
    ------
    TypeError
        If `adata` is not an instance of `AnnData`.
    ValueError
        If `layer` is not found in `adata.layers`.
        If `dim_space` is not "obs" or "var".
        If `meta_data_mask_column_name` is not found in `adata.var`.
    TypeError
        If `meta_data_mask_column_name` is not of boolean dtype.
    """
    logger.debug("Checking inputs for dimensionality reduction")
    # check inputs
    if not isinstance(adata, (ad.AnnData)):
        raise TypeError(f"Data should be AnnData object, got {type(adata)}")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer {layer} not found in AnnData object, available layers: {adata.layers.keys()}")

    if dim_space not in ["obs", "var"]:
        raise ValueError(f"dim_space should be either 'obs' or 'var', got {dim_space}")

    if meta_data_mask_column_name is not None:
        if meta_data_mask_column_name not in adata.var.columns:
            raise ValueError(f"Column {meta_data_mask_column_name} not found in data.var")
        if adata.var[meta_data_mask_column_name].dtype.kind != "b":
            raise TypeError(
                f"adata.var['{meta_data_mask_column_name}'] must be of boolean dtype, but it's {adata.var[meta_data_mask_column_name].dtype}."
            )


def _get_pca_attr_key_names(adata: ad.AnnData, dim_space: str, embbedings_name: str | None) -> tuple[str, str, str]:
    """Get the attribute names for PCA results based on the dimension space and embeddings name.

    Parameters
    ----------
    dim_space: str
        The dimension to project PCA on. Can be either "obs" for sample projection or "var" for feature projection.
    embbedings_name: str, optional
        The key under which to store the PCA results in adata.

    Returns
    -------
    tuple[str, str, str, str, str, str]
        A tuple containing the attribute and the key names for PCA coordinates, loadings, and variance.
    """
    # define key names for storing PCA results
    pca_coors_key = f"X_pca_{dim_space}" if embbedings_name is None else embbedings_name
    loadings_key = f"PCs_{dim_space}" if embbedings_name is None else embbedings_name
    variance_key = f"variance_pca_{dim_space}" if embbedings_name is None else embbedings_name

    # set attr names names for storing PCA results
    pca_coors_attr = "obsm" if dim_space == "obs" else "varm"
    loadings_attr = "varm" if dim_space == "obs" else "obsm"
    variance_attr = "uns"

    if pca_coors_key in getattr(adata, pca_coors_attr):
        logger.warning(f"Overwriting existing PCA coordinates in {pca_coors_attr}['{pca_coors_key}']")
    if loadings_key in getattr(adata, loadings_attr):
        logger.warning(f"Overwriting existing PCA loadings in {loadings_attr}['{loadings_key}']")
    if variance_key in getattr(adata, variance_attr):
        logger.warning(f"Overwriting existing PCA variance in {variance_attr}['{variance_key}']")

    return pca_coors_attr, pca_coors_key, loadings_attr, loadings_key, variance_attr, variance_key


def pca(
    adata: ad.AnnData,
    layer: str | None = None,
    dim_space: str = "obs",
    embbedings_name: str | None = None,
    n_comps: int | None = None,
    meta_data_mask_column_name: str | None = None,
    **pca_kwargs: dict | None,
) -> ad.AnnData | np.ndarray:
    """Principal component analysis :cite:p:`Pedregosa2011`.

    Computes PCA coordinates, loadings and variance decomposition. The passed adata will be changed as a result to include the pca calculations.
    depending on the `dim_space` parameter, the PCA result is dimensional reduction projection of samples (`obs`) or of features (`var`).
    After PCA, the updated adata object will include `adata.obsm` layer for the PCA coordinates,`adata.varm` layer (for PCA feature loadings),
    and `adata.uns` layer (for PCA variance decomposition) for PCA done on the feature space.
    For PCA done on the sample space, the PCA coordinates will be stored in `adata.varm`, the PCA loadings in `adata.obsm`, and the variance decomposition in `adata.uns`.
    Uses the implementation of Scanpy, which in turn uses implementation of
    *scikit-learn* :cite:p:`Pedregosa2011`.

    Parameters
    ----------
    adata: ad.AnnData
        The (annotated) data matrix of shape `n_obs` X `n_vars`.
        Rows correspond to cells and columns to genes.
    layer: str, optional (default: None)
        If provided, which element of layers to use for PCA.
        If None, the `.X` attribute of `adata` is used.
    dim_space: str, optional (default: "obs")
        The dimension to project PCA on. Can be either "obs" (default) for
        sample projection or "var" for feature projection.
    embbedings_name: str, optional
        The key under which to store the PCA results in adata
    n_comps: int, optional (default: 50)
        Number of principal components to compute. Defaults to 50, or 1 - minimum
        dimension size of selected representation.
    meta_data_mask_column_name: str, optional (default: None)
        If provided, the colname in `adata.var` to use as a mask for
        the features to be used in PCA. This is useful for running PCA with the
        core proteome as "mask_var" to remove nan values. Must be of boolean dtype.
        If None, all features are used (data should not include NaNs!).
    **pca_kwargs: dict, optional
        Additional keyword arguments for the :func:`scanpy.pp.pca` By default None.

    Returns
    -------
    (as output from the scanpy.pp.pca function)
    unless changed in the kwargs passed on to scanpy, an updated `AnnData` object.
    Sets the following fields:
    for `dim_space='obs'` (sample projection):
    `.obsm['X_pca_obs' | embbedings_name]` : :class:`~scipy.sparse.csr_matrix` | :class:`~scipy.sparse.csc_matrix` | :class:`~numpy.ndarray` (shape `(adata.n_obs, n_comps)`)
        PCA representation of data.
    `.varm['PCs_obs' | embbedings_name]` : :class:`~numpy.ndarray` (shape `(adata.n_vars, n_comps)`)
        The principal components containing the loadings.
    `.uns['variance_pca_obs' | embbedings_name]['variance_ratio']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Ratio of explained variance.
    `.uns['variance_pca_obs' | embbedings_name]['variance']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Explained variance, equivalent to the eigenvalues of the
        covariance matrix.

    for `dim_space='var'` (sample projection):
    `.varm['X_pca_var' | embbedings_name]` : :class:`~scipy.sparse.csr_matrix` | :class:`~scipy.sparse.csc_matrix` | :class:`~numpy.ndarray` (shape `(adata.n_obs, n_comps)`)
        PCA representation of data.
    `.obsm['PCs_var' | embbedings_name]` : :class:`~numpy.ndarray` (shape `(adata.n_vars, n_comps)`)
        The principal components containing the loadings.
    `.uns['variance_pca_var' | embbedings_name]['variance_ratio']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Ratio of explained variance.
    `.uns['variance_pca_var' | embbedings_name]['variance']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Explained variance, equivalent to the eigenvalues of the
        covariance matrix.
    """
    logger.info("computing PCA")
    pca_kwargs = pca_kwargs or {}

    _check_inputs_for_dim_reduction(
        adata=adata, layer=layer, dim_space=dim_space, meta_data_mask_column_name=meta_data_mask_column_name
    )

    # get the matrix to run PCA on
    adata_sub = adata[:, adata.var[meta_data_mask_column_name]] if meta_data_mask_column_name is not None else adata
    data_for_pca = adata_sub.layers[layer].copy() if layer is not None else adata_sub.X.copy()
    data_for_pca = (
        data_for_pca.T if dim_space == "var" else data_for_pca
    )  # transpose if PCA is done on the feature space

    # run PCA
    pca_res = sc.pp.pca(data_for_pca, return_info=True, n_comps=n_comps, **pca_kwargs)

    # get the attribute and key names for storing PCA results
    pca_coors_attr, pca_coors_key, loadings_attr, loadings_key, variance_attr, variance_key = _get_pca_attr_key_names(
        adata=adata, dim_space=dim_space, embbedings_name=embbedings_name
    )

    if meta_data_mask_column_name is not None:  # PCs
        n_pcs = pca_res[0].shape[1]
        mask = np.where(adata.var[meta_data_mask_column_name].values)[0]

        if dim_space == "var":
            # PC coordinates of the features used in PCA (NA to all features not used in PCA)
            pc_mat = np.full((adata.n_vars, n_pcs), np.nan)
            pc_mat[mask, :] = pca_res[0].copy()

            # sample loading of the samples used in PCA
            loadings_mat = pca_res[1].T.copy()

        else:  # dim_space == "obs":
            # PC coordinates of the samples used in PCA
            pc_mat = pca_res[0].copy()

            # feature loading of the features used in PCA (NA to all features not used in PCA)
            loadings_mat = np.full((adata.n_vars, n_pcs), np.nan)
            loadings_mat[mask, :] = pca_res[1].T.copy()
    else:  # no mask
        pc_mat = pca_res[0].copy()
        loadings_mat = pca_res[1].T.copy()

    getattr(adata, pca_coors_attr)[pca_coors_key] = pc_mat
    getattr(adata, loadings_attr)[loadings_key] = loadings_mat
    getattr(adata, variance_attr)[variance_key] = {
        "variance_ratio": pca_res[2].copy(),  # Ratio of explained variance (n_comp)
        "variance": pca_res[3].copy(),  # Explained variance (n_comp)
    }

    return adata
