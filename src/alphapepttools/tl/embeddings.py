import logging
from collections.abc import Iterable
from typing import Literal

import anndata as ad
import numpy as np
import scanpy as sc
from bpca import BPCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_inputs_for_dim_reduction(
    adata: ad.AnnData, layer: str | None, dim_space: str, meta_data_mask_column_name: str | None
) -> None:
    """Check inputs for PCA and other dimensionality reduction methods.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` X `n_vars`.
    layer
        Layer name to check. If None, default to `adata.X`
    dim_space
        Must be "obs" or "var". ValueError otherwise.
    meta_data_mask_column_name
        Colname to check in `adata.var`. Must be of boolean dtype.

    Raises
    ------
    TypeError
        If adata is not an AnnData object or if meta_data_mask_column_name exists but is not boolean dtype.
    ValueError
        If layer is not found in adata.layers, dim_space is not 'obs' or 'var', or meta_data_mask_column_name is not found in adata.var.
    TypeError
        If adata.var[metadata_mask_column_name] is not boolean dtype
    ValueError
        If adata.var[metadata_mask_column_name] does not exist

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


def _prepare_pca_data(
    adata: ad.AnnData,
    layer: str | None = None,
    var_mask: Iterable[bool] | None = None,
    dim_space: Literal["obs", "var"] = "obs",
) -> np.ndarray:
    """Extract data for PCA in correct orientation

    Parameters
    ----------
    adata
        AnnData object (obs x var)
    layer
        Layer in anndata object to consider. If `None` uses `adata.X`.
    var_mask
        Boolean mask indicating whether feature should be considered for PCA or not
    dim_space
        PCA projection space. Either "obs" (project observations) or "var" (project features).

    Returns
    -------
    Array with dimensions `(obs, var)` if `dim_space == "obs"`, or `(var, obs)` if
    `dim_space == "var"`. The var dimension includes only features for which `var_mask`
    is True.
    """
    adata_sub = adata[:, var_mask] if var_mask is not None else adata
    data_for_pca = adata_sub.layers[layer].copy() if layer is not None else adata_sub.X.copy()

    # Transpose if PCA is done on the feature space
    return data_for_pca.T if dim_space == "var" else data_for_pca


def _store_pca_results(
    adata: ad.AnnData,
    pca_res: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
    default_coords_prefix: str,
    default_loadings_prefix: str,
    default_uns_prefix: str,
    dim_space: Literal["obs", "var"],
    embeddings_name: str | None,
    meta_data_mask_column_name: str | None,
) -> ad.AnnData:
    """Store PCA results (coordinates, loadings, and variance) in the appropriate AnnData attributes (.obsm, .varm, .uns)

    Per default, keys are generated in the form `{default_<>_key}_{dim_space}` and added to the respective
    anndata attributes. `embeddings_name` overwrites the defaults in which case all added keys will be called
    `embeddings_name`.

    Parameters
    ----------
    adata
        The AnnData object to update.
    pca_res
        PCA result tuple (coordinates, loadings, variance_ratio, variance).
    default_coords_prefix
        Default prefix of coordinates. Overwritten by `embeddings_name`
    default_loadings_prefix
        Default prefix of loadings. Overwritten by `embeddings_name`
    default_uns_prefix
        Default prefix of metadata in `adata.uns`. Overwritten by `embeddings_name`
    dim_space
        Either "obs" or "var", indicating the PCA projection space.
    embeddings_name
        Custom key name for storing PCA results, used in all attributes. If `None`, keys are {default_<>_prefix}_{dim_space}
    meta_data_mask_column_name
        Column name in adata.var used as a boolean mask for features. If None, all features are used.

    Returns
    -------
    The updated AnnData object with PCA results added to `adata.obsm`, `adata.varm`, and `adata.uns` attributes
    """
    # get key names for storing PCA results
    if embeddings_name is None:
        pca_coords_key = f"{default_coords_prefix}_{dim_space}"
        loadings_key = f"{default_loadings_prefix}_{dim_space}"
        variance_key = f"{default_uns_prefix}_{dim_space}"
    else:
        pca_coords_key = embeddings_name
        loadings_key = embeddings_name
        variance_key = embeddings_name

    # check if PCA was run for all features or only for a subset
    if meta_data_mask_column_name is None:
        pc_mat = pca_res[0].copy()
        loadings_mat = pca_res[1].T.copy()
    else:
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
            # feature loading of the features used in PCA (nan values for all features NOT used in PCA)
            loadings_mat = np.full((adata.n_vars, n_pcs), np.nan)
            loadings_mat[mask, :] = pca_res[1].T.copy()

    if dim_space == "obs":
        coords_dict, loadings_dict = adata.obsm, adata.varm
        coords_location, loadings_location = "obsm", "varm"

    else:  # dim_space == "var"
        coords_dict, loadings_dict = adata.varm, adata.obsm
        coords_location, loadings_location = "varm", "obsm"

    # overwrite existing keys if they exist
    if variance_key in adata.uns:
        logger.warning(f"Overwriting existing PCA variance in uns.['{variance_key}']")
    if pca_coords_key in coords_dict:
        logger.warning(f"Overwriting existing PCA coordinates {coords_location}.['{pca_coords_key}']")
    if loadings_key in loadings_dict:
        logger.warning(f"Overwriting existing PCA loadings {loadings_location}.['{loadings_key}']")

    # store PCA results in locations
    coords_dict[pca_coords_key] = pc_mat
    loadings_dict[loadings_key] = loadings_mat
    adata.uns[variance_key] = {
        "variance_ratio": pca_res[2].copy(),  # Ratio of explained variance (n_comp)
        "variance": pca_res[3].copy() if pca_res[3] is not None else None,  # Explained variance (n_comp)
    }

    return adata


def pca(
    adata: ad.AnnData,
    layer: str | None = None,
    dim_space: str = "obs",
    embeddings_name: str | None = None,
    n_comps: int | None = None,
    meta_data_mask_column_name: str | None = None,
    **pca_kwargs: dict | None,
) -> ad.AnnData | np.ndarray:
    """Principal component analysis :cite:p:`Pedregosa2011`.

    Computes PCA coordinates, loadings and variance decomposition. The passed adata will be changed as a result to include the pca calculations.
    depending on the `dim_space` parameter, the PCA result is dimensionality reduction projection of samples (`obs`) or of features (`var`).
    After PCA, the updated adata object will include `adata.obsm` layer for the PCA coordinates,`adata.varm` layer (for PCA feature loadings),
    and `adata.uns` layer (for PCA variance decomposition) for PCA done on the feature space.
    For PCA done on the sample space, the PCA coordinates will be stored in `adata.varm`, the PCA loadings in `adata.obsm`, and the variance decomposition in `adata.uns`.
    Uses the implementation of Scanpy, which in turn uses implementation of
    *scikit-learn* :cite:p:`Pedregosa2011`.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` X `n_vars`.
        Rows correspond to samples and columns to features.
    layer
        If provided, which element of layers to use for PCA.
        If None, the `.X` attribute of `adata` is used.
    dim_space
        The dimension to project PCA on. Can be either "obs" (default) for
        sample projection or "var" for feature projection.
    embeddings_name
        If provided, this will be used as the key under which to store the PCA results in
        `adata.obsm`, `adata.varm`, and `adata.uns` (see Returns).
        If None, the default keys will be used:
        - For `dim_space='obs'`: `X_pca_obs` for PC coordinates, `PCs_obs` for the feature loadings, `variance_pca_obs` for the variance.
        - For `dim_space='var'`: `X_pca_var` for PC corrdinates, `PCs_var` for the sample loadings, `variance_pca_var` for the variance.
        If provided, the keys will be `embeddings_name` for all three data frames.
    n_comps
        Number of principal components to compute. Defaults to 50, or 1 - minimum
        dimension size of selected representation.
    meta_data_mask_column_name
        If provided, the colname in `adata.var` to use as a mask for
        the features to be used in PCA. This is useful for running PCA with the
        core proteome as "mask_var" to remove nan values. Must be of boolean dtype.
        If None, all features are used (data should not include NaNs!).
    **pca_kwargs
        Additional keyword arguments for the :func:`scanpy.pp.pca` By default None.

    Returns
    -------
    (as output from the scanpy.pp.pca function)
    unless changed in the kwargs passed on to scanpy, an updated `AnnData` object.
    Sets the following fields:
    for `dim_space='obs'` (sample projection):
    `.obsm['X_pca_obs' | embeddings_name]` : :class:`~scipy.sparse.csr_matrix` | :class:`~scipy.sparse.csc_matrix` | :class:`~numpy.ndarray` (shape `(adata.n_obs, n_comps)`)
        PCA representation of data.
    `.varm['PCs_obs' | embeddings_name]` : :class:`~numpy.ndarray` (shape `(adata.n_vars, n_comps)`)
        The principal components containing the loadings.
    `.uns['variance_pca_obs' | embeddings_name]['variance_ratio']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Ratio of explained variance.
    `.uns['variance_pca_obs' | embeddings_name]['variance']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Explained variance, equivalent to the eigenvalues of the
        covariance matrix.

    for `dim_space='var'` (sample projection):
    `.varm['X_pca_var' | embeddings_name]` : :class:`~scipy.sparse.csr_matrix` | :class:`~scipy.sparse.csc_matrix` | :class:`~numpy.ndarray` (shape `(adata.n_obs, n_comps)`)
        PCA representation of data.
    `.obsm['PCs_var' | embeddings_name]` : :class:`~numpy.ndarray` (shape `(adata.n_vars, n_comps)`)
        The principal components containing the loadings.
    `.uns['variance_pca_var' | embeddings_name]['variance_ratio']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Ratio of explained variance.
    `.uns['variance_pca_var' | embeddings_name]['variance']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Explained variance, equivalent to the eigenvalues of the
        covariance matrix.

    Examples
    --------
    Run PCA using a metadata mask to select core proteins:

    .. code-block:: python

        import anndata as ad
        import pandas as pd
        import numpy as np
        import alphapepttools as at

        # Create a 5x5 dataset where 4 proteins are core (no missing values)
        X = np.array(
            [
                [10.5, 12.3, 11.8, 9.2, np.nan],  # Sample 1
                [11.2, 13.1, 12.5, 10.1, 7.5],  # Sample 2
                [9.8, 11.9, 10.2, 8.9, np.nan],  # Sample 3
                [12.1, 14.2, 13.3, 11.3, 8.2],  # Sample 4
                [10.9, 12.7, 11.5, 9.8, np.nan],  # Sample 5
            ]
        )

        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame({"sample": ["S1", "S2", "S3", "S4", "S5"]}),
            var=pd.DataFrame(
                {
                    "protein": ["P1", "P2", "P3", "P4", "P5"],
                    "is_core": [True, True, True, True, False],  # First 4 are core proteins
                }
            ),
        )

        # Run PCA on feature space using only core proteins
        at.tl.pca(adata, meta_data_mask_column_name="is_core", n_comps=2, dim_space="var")

        # The PCA results are now stored in the AnnData object:
        # adata.varm['X_pca_var'] - PCA coordinates for each protein (5 x 2)
        # adata.obsm['PCs_var'] - Sample loadings (5 x 2)
        # adata.uns['variance_pca_var'] - Variance explained by each PC

        # To get the PCA embedding of proteins in the reduced space:
        protein_pca_coords = adata.varm["X_pca_var"]
        # First 4 proteins have coordinates, P5 has NaN (not used in PCA)

        # To project samples into the PC space:
        sample_loadings = adata.obsm["PCs_var"]

        # To see variance explained by each component:
        variance_ratio = adata.uns["variance_pca_var"]["variance_ratio"]

    """
    logger.info("computing PCA")

    _check_inputs_for_dim_reduction(
        adata=adata, layer=layer, dim_space=dim_space, meta_data_mask_column_name=meta_data_mask_column_name
    )

    var_mask = adata.var[meta_data_mask_column_name] if meta_data_mask_column_name is not None else None
    data_for_pca = _prepare_pca_data(adata=adata, layer=layer, var_mask=var_mask, dim_space=dim_space)

    # run PCA
    pca_res = sc.pp.pca(data_for_pca, return_info=True, n_comps=n_comps, **pca_kwargs)

    return _store_pca_results(
        adata=adata,
        pca_res=pca_res,
        dim_space=dim_space,
        embeddings_name=embeddings_name,
        meta_data_mask_column_name=meta_data_mask_column_name,
        default_coords_prefix="X_pca",
        default_loadings_prefix="PCs",
        default_uns_prefix="variance_pca",
    )


def _run_bpca(
    data_for_bpca: np.ndarray, n_components: int, **bpca_kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    """Run Bayesian Principal Component Analysis

    Parameters
    ----------
    data_for_bpca
        Data of shape (dim0, dim1)
    n_components
        Number of components
    **bpca_kwargs
        Passed to :class:`BPCA`

    Returns
    -------
    Tuple of numpy arrays
        - usage: BPCA factor usage (dim0, n_components)
        - loadings: BPCA factor loadings (n_components, dim1)
        - variance_ratio: Fraction of variance explained (n_components,)
        - eigenvalues: None as `BPCA` does not compute the eigenvalues of the covariance matrix. Returned for compatibility with :func:`alphapepttools.tl.pca`.
    """
    bpca = BPCA(n_components=n_components, sort_components=True, **bpca_kwargs)
    usage = bpca.fit_transform(data_for_bpca)
    loadings = bpca.components_
    explained_variance_ratio = bpca.explained_variance_ratio_

    return (usage, loadings, explained_variance_ratio, None)


def bpca(
    adata: ad.AnnData,
    layer: str | None = None,
    dim_space: str = "obs",
    embeddings_name: str | None = None,
    n_comps: int = 50,
    meta_data_mask_column_name: str | None = None,
    **bpca_kwargs,
) -> ad.AnnData:
    """Bayesian Principal Component Analysis

    Bayesian implementation of PCA that explicitly supports missing values. Computes latent space coordinates, loadings and variance decomposition.

    The dimensionality-reduced representation can be computed either for samples (`dim_space="obs"`) or for features (`dim_space="var"`).
    Depending on the chosen `dim_space`, the BPCA results are stored in different AnnData containers.

    - For BPCA computed in feature space (`dim_space='var'`)
      The low-dimensional coordinates are stored in `adata.obsm`, the feature loadings in `adata.varm`, and the variance decomposition in `adata.uns`.
    - For BPCA computed in sample space (`dim_space='obs'`)
      The coordinates are stored in `adata.varm`, the loadings in `adata.obsm`, and the variance decomposition in `adata.uns`.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` X `n_vars`.
        Rows correspond to samples and columns to features.
    layer
        If provided, which element of layers to use for PCA.
        If None, the `.X` attribute of `adata` is used.
    dim_space
        The dimension to project PCA on. Can be either "obs" (default) for
        sample projection or "var" for feature projection.
    embeddings_name
        If provided, this will be used as the key under which to store the PCA results in
        `adata.obsm`, `adata.varm`, and `adata.uns` (see Returns).
        If None, the default key `"BPCA"` is used for storing results in `adata.obsm`, `adata.varm`, and `adata.uns`.
    n_comps
        Number of principal components to compute. Defaults to `min(50, n_obs, n_var)`
    meta_data_mask_column_name
        If provided, the colname in `adata.var` to use as a mask for
        the features to be used in PCA. This is useful for running PCA with the
        core proteome as "mask_var" to remove nan values. Must be of boolean dtype.
    **bpca_kwargs
        Additional keyword arguments to :class:`bpca.BPCA`. By default None.

    Returns
    -------
    Sets the following fields:
    for `dim_space='obs'` (sample projection):
    `.obsm['BPCA' | embeddings_name]` : :class:`~numpy.ndarray` (shape `(adata.n_obs, n_comps)`)
        BPCA representation of data.
    `.varm['BPCA' | embeddings_name]` : :class:`~numpy.ndarray` (shape `(adata.n_vars, n_comps)`)
        The principal components containing the loadings.
    `.uns['BPCA' | embeddings_name]['variance_ratio']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Ratio of explained variance.

    for `dim_space='var'` (feature projection):
    `.varm['BPCA' | embeddings_name]` : :class:`~numpy.ndarray` (shape `(adata.n_obs, n_comps)`)
        BPCA representation of data.
    `.obsm['BPCA' | embeddings_name]` : :class:`~numpy.ndarray` (shape `(adata.n_vars, n_comps)`)
        The principal components containing the loadings.
    `.uns['BPCA' | embeddings_name]['variance_ratio']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Ratio of explained variance.

    Notes
    -----
    For complete data, BPCA converges to the standard PCA solution, but typically requires substantially more computation due to iterative model fitting.

    BPCA assumes additive, homoscedastic Gaussian noise in the observed data. After appropriate normalization and log-transformation, this assumption is
    often a reasonable approximation for quantitative proteomics data, but may still be violated for features with extreme missingness or low signal intensity.
    In practice, filtering features with very high missingness prior to BPCA can improve numerical stability and interpretability.

    Example
    -------
    As the BPCA method supports missing values, you can directly run the dimensionality reduction on the log-transformed dataset.

    .. code-block:: python

        path = at.data.get_data("bader2020_full_diann")
        adata = at.io.read_psm_table(path, search_engine="diann")

        # Optional: Remove features with little data support
        at.pp.filter_data_completeness(
            adata=adata,
            max_missing=0.25,
            action="drop",
        )

        # BPCA expects a normal noise model, thus the data should be log-transformed
        at.pp.nanlog(adata)

        at.tl.bpca(adata)


    References
    ----------
    - :cite:p:`Oba.2003`
    - :cite:p:`Bishop.1998`

    See Also
    --------
    :class:`bpca.BPCA`
    """
    _check_inputs_for_dim_reduction(
        adata=adata, layer=layer, dim_space=dim_space, meta_data_mask_column_name=meta_data_mask_column_name
    )

    var_mask = adata.var[meta_data_mask_column_name] if meta_data_mask_column_name is not None else None
    data_for_bpca = _prepare_pca_data(adata=adata, layer=layer, var_mask=var_mask, dim_space=dim_space)

    pca_res = _run_bpca(data_for_bpca=data_for_bpca, n_components=n_comps, **bpca_kwargs)

    return _store_pca_results(
        adata=adata,
        pca_res=pca_res,
        dim_space=dim_space,
        embeddings_name=embeddings_name,
        meta_data_mask_column_name=meta_data_mask_column_name,
        default_coords_prefix="X_bpca",
        default_loadings_prefix="PCs_bpca",
        default_uns_prefix="variance_bpca",
    )
