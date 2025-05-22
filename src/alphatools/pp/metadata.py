import logging

import anndata as ad
import numpy as np

# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_core_proteome_mask(adata: ad.AnnData, layer: str, new_column_name: str = "is_core") -> ad.AnnData:
    """Adds a column of booleans to the adata.var table.

    indicates whether the protein appears in 100% of the samples (TRUE) or not (FALSE).
    this is useful for running PCA with the core proteome as "mask_var".

    Parameters
    ----------
    adata: ad.AnnData
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
    if not isinstance(adata, (ad.AnnData)):
        raise TypeError("Data should be AnnData object")
    if layer and layer not in adata.layers:
        raise ValueError(f"Layer {layer} not found in AnnData object")
    mat = adata.X if layer is None else adata.layers[layer]

    # indicate which proteins are in the core proteome
    if new_column_name in adata.var.columns:
        logger.warning(f"'{new_column_name}' exists in data.var, overwriting")

    adata.var[new_column_name] = ~np.isnan(mat).any(axis=0)
    logger.info(f"'{new_column_name}' column added to data.var to classify core proteins")

    return adata
