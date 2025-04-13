"""Principal component regression"""

import warnings

import anndata as ad
import numpy as np
import scanpy as sc  # TODO: align with alphatools implementation
from scipy.stats import pearsonr


def _pcr(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    """Weighted mean of explained variance by covariate per component

    Total variance explained by covariate in passed componentes. The variance explained by the covariate
    in each component is approximated with the coefficient of determination, assuming a linear relationship,
    and weighted by the explained variance

    Parameters
    ----------
    x
        N observations x 1 (covariate, encoded). Values of covariate
    y
        N observations x C components. PCA usage matrix.
    explained_variance
        C components x 1. Explained variance per component/weighting factor

    Returns
    -------
    Weighted mean variance explained by covariate over passed components
    """
    # TODO: Allow for multi-dimensional covariate values
    return sum(
        var_explained * np.square(pearsonr(x, yi).statistic) for yi, var_explained in zip(y.T, weight, strict=True)
    )


def principal_component_regression(
    adata: ad.AnnData,
    covariate: str,
    n_components: int | None = None,
    pca_key: str = "pca",
    pca_key_uns: str | None = None,
) -> float:
    r"""Compute principal component regression

    Computes correlation between covariate of interest C and principal components 1, ..., n_components
    The total variance explained in the latent space is derived by component-wise computation of the
    variance explained and multiplying with the total variance explained by this commponent Var(PC_n)

    .. math::

        PCR = \sum_{n=1}^{N}{PCC(C, PC_n)^2 \cdot Var(PC_n)}

    See Also
    --------
    - Luecken, M.D., Büttner, M., Chaichoompu, K. et al. Benchmarking atlas-level data integration in single-cell genomics. Nat Methods 19, 41-50 (2022). https://doi.org/10.1038/s41592-021-01336-8
    """
    # Extract principal component usage matrix from adata.obsm
    if pca_key not in adata.obsm:
        warnings.warn(f"Run PCA as pca key {pca_key} was not found")
        sc.pp.pca(adata, random_state=42)

    # The results of the PCA are usually stored in adata.uns[<pca_key>]
    # but in case the name differs (e.g. obsm: X_pca/.uns: pca), add the possibility
    # to manually specify the key
    if pca_key_uns is None:
        pca_key_uns = pca_key

    pca_usage = adata.obsm[pca_key][:n_components]
    pca_explained_variance = adata.uns[pca_key]["variance_ratio"][:n_components]

    covariate_values = adata.obs[covariate]

    # Covariate values should be numeric
    # TODO: - make one-hot encoded
    if covariate_values.dtype.name == "category":
        covariate_values = covariate_values.cat.codes

    return _pcr(x=covariate_values, y=pca_usage, weight=pca_explained_variance)
