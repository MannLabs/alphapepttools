"""Principal component regression"""

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _verify_keys__principal_component_regression(
    adata: ad.AnnData, covariate: str, pca_key: str, pca_uns_key: str
) -> None:
    """Validate keys required for PCR

    Raises
    ------
    KeyError
        For missing keys
    """
    if pca_key not in adata.obsm:
        raise KeyError(
            f"Key `pca_key={pca_key}` was not found in `adata.obsm`. Run `alphatools.pp.pca` first or specify correct key."
        )

    if pca_uns_key not in adata.uns:
        raise KeyError(
            f"Key `pca_key_uns={pca_uns_key}` was not found in `adata.uns`. Run `alphatools.pp.pca` first or specify correct key."
        )

    if covariate not in adata.obs:
        raise KeyError(f"Column `{covariate}` not found in `adata.obs`")


def _pcr(principal_component_embeddings: np.ndarray, covariate: np.ndarray, explained_variance: np.ndarray) -> float:
    """Weighted mean of explained variance

    Parameters
    ----------
    principal_component_embeddings
        PCA embeddings matrix (N observations x C components).
    covariate
        Values of covariate (N observations x 1 | L levels (covariate, encoded)).
    explained_variance
        Explained variance per component/weighting factor (C components x 1).

    Returns
    -------
    Explained variance of covariate over C components assuming a linear relationship
    """
    # Transpose from (samples, PCs) to (PCs, samples) to easily iterate through PCs
    principal_component_embeddings = principal_component_embeddings.T

    return sum(
        var_explained * LinearRegression(fit_intercept=True).fit(covariate, pc_i).score(covariate, pc_i)
        for pc_i, var_explained in zip(principal_component_embeddings, explained_variance, strict=True)
    )


def principal_component_regression(
    adata: ad.AnnData,
    covariate: str,
    n_components: int | None = None,
    pca_key: str = "X_pca",
    pca_key_uns: str = "pca",
) -> float:
    r"""Compute principal component regression

    Computes correlation between covariate of interest C and principal components 1, ..., n_components
    The total variance explained in the latent space is derived by component-wise computation of the
    variance explained and multiplying with the total variance explained by this commponent Var(PC_n)

    .. math::

        \mathrm{PCR} = \sum_{n=1}^{N}{\left( \mathrm{PCC}(C, PC_n)^2 \cdot \mathrm{Var}(PC_n) \right)}

    Parameters
    ----------
    adata
        :class:`ad.AnnData` object
    covariate
        Covariate of interest as column in `adata.obs`. For continuous covariates, the pearson correlation coefficient (PCC) is computed between covariate and
        principal component. Categorical covariates (`dtype=category`) are one hot encoded.
    n_components
        Number of principal components to consider. If `None`, uses all available components.
    pca_key
        Key in `adata.obsm` that stores PCA embeddings.
    pca_key_uns
        Key in `adata.uns` that stores information on the PCA.

    Returns
    -------
    Principal component regression
        Aggregated explained variance of covariate in Principal Component Space

    Raises
    ------
    KeyError
        For missing keys
    TypeError
        If `covariate` dtype is not numeric or categorical

    Usage
    -----

    .. code-block:: python

        import alphatools as at

        at.pp.pca(adata)
        at.metrics.principal_component_regression(adata, covariate="batch")

        # With custom PCA keys
        at.pp.pca(adata, layer="layer1", key_added="pca_layer1")
        at.metrics.principal_component_regression(
            adata, covariate="batch", pca_key="pca_layer1", pca_uns_key="pca_layer1"
        )

    See Also
    --------
    - Luecken, M.D., Büttner, M., Chaichoompu, K. et al. Benchmarking atlas-level data integration in single-cell genomics. Nat Methods 19, 41-50 (2022). https://doi.org/10.1038/s41592-021-01336-8
    - Büttner, M., Miao, Z., Wolf, F.A. et al. A test metric for assessing single-cell RNA-seq batch correction. Nat Methods 16, 43-49 (2019). https://doi.org/10.1038/s41592-018-0254-1
    """
    _verify_keys__principal_component_regression(
        adata, covariate_key=covariate, pca_key=pca_key, pca_uns_key=pca_key_uns
    )

    pca_embeddings = adata.obsm[pca_key]
    explained_variance = adata.uns[pca_key_uns]["variance_ratio"]

    if n_components is not None:
        pca_embeddings = pca_embeddings[:, :n_components]
        explained_variance = explained_variance[:n_components]

    y = adata.obs[covariate]
    if pd.api.types.is_numeric_dtype(y):
        y = y.to_numpy().reshape(-1, 1)
    elif pd.api.types.is_categorical_dtype(y):
        y = pd.get_dummies(y).to_numpy()
    else:
        raise TypeError(f"Dtype of column {y.dtype} not supported. Must be numeric or categorical")

    return _pcr(pca_embeddings, y, explained_variance)
