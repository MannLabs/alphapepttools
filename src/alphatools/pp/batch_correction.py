import logging

import anndata as ad
import numpy as np
import scanpy

from alphatools.pp.impute import impute_gaussian

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scanpy_pycombat(
    adata: ad.AnnData,
    batch: str,
    imputer: str = "gaussian",
    *,
    return_imputed: bool = False,
) -> ad.AnnData:
    """Apply batch correction using scanpy's implementation of pyComBat

    Correct for the batch effect of a categorical covariate using an empirical
    Bayes framework as implemented in the pyComBat function of scanpy. Since the
    underlying function requires a complete data matrix without NaN values, an imputation
    strategy is needed to perform pycombat on matrices with missing values.
    If `return_imputed` is False (default), missing values will be propagated to the
    batch-corrected output.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix, where rows are cells and columns are features.
    batch : str
        Name of the batch feature in obs, the variation associated with this feature will be corrected
    imputer : str
        Method to impute missing values. Current options are: 'gaussian'.
    return_imputed : bool
        Whether to return imputed values or leave them as NaN. Default is False.

    Returns
    -------
    adata : anndata.AnnData
        Annotated data matrix with batch correction applied.
        If `return_imputed` is False, any NaN values from the input data will remain NaN in the output data.

    """
    logging.info(f" |-> Apply pyComBat to correct for {batch}")

    # always copy for now, implement inplace later if needed
    adata = adata.copy()

    # pycombat crashes when the input data contains NaNs
    # missing values are always imputed and optionally returned
    impute_values = False
    na_mask = np.isnan(adata.X)
    if np.isnan(adata.X).any():
        logging.info(
            f" |-> Data contains {np.isnan(adata.X).sum()} nans. Imputing values for calculations with {imputer}."
        )
        impute_values = True

        if imputer == "gaussian":
            adata = impute_gaussian(adata)
        else:
            raise ValueError(f"Imputer {imputer} not supported. Use 'gaussian'.")

    # 'batch' column must not contain NaNs. If it does, simply add a "NA" batch
    if adata.obs[batch].isna().sum() > 0:
        adata.obs[batch] = adata.obs[batch].fillna("NA")

    # If any level of the to-correct data has only one level, drop it
    batch_level_count = adata.obs.groupby(batch)[batch].transform("count")

    if any(batch_level_count == 1):
        logging.info(f"There are single-sample batches for {batch}. Dropping these samples.")
        adata = adata[batch_level_count > 1, :].copy()
        na_mask = na_mask[batch_level_count > 1, :]

    # batch correct; apparently pyCombat from scanpy sets everything to nan if there is a batch that contains only one cell
    # i.e. if a sample only occurs once per plate, everything fails and we get all nans
    # this is a known issue, but it is not clear how to fix this https://github.com/scverse/scanpy/issues/1175
    # for now, we will just catch the error and report it.
    if any(adata.obs[batch].value_counts() == 1):
        logging.warning(
            f" |-> tools.py/scanpy_pyComBat: for batch correction: at least one batch of {batch} contains only one single sample. This causes scanpy.pp.combat to fail and return all nans, therefore batch correction can not be performed."
        )
        raise ValueError("At least one batch contains only one single sample.")
    scanpy.pp.combat(adata, key=batch, inplace=True)

    # return data with imputed values if requested
    if not return_imputed and impute_values:
        logging.info(" |-> Imputed values will not be returned.")
        adata.X[na_mask] = np.nan

    return adata
