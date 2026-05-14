import logging

import anndata as ad
import numpy as np
import scanpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def coerce_nans_to_batch(
    adata: ad.AnnData,
    batch: str,
) -> ad.AnnData:
    """Coerce NaN values in a batch column to a single "NA" batch.

    Parameters
    ----------
    adata
        Annotated data matrix, where rows are cells and columns are features
    batch
        Name of the batch feature in obs, the variation associated with this feature will be corrected

    Returns
    -------
    Annotated data matrix with NaN values in the batch column replaced by "NA"

    Examples
    --------
    .. code-block:: python

        import anndata as ad
        import pandas as pd
        import numpy as np
        from alphapepttools.pp.batch_correction import coerce_nans_to_batch

        # Create a sample AnnData object with NaN values in the batch column
        data = np.random.rand(5, 3)
        obs = pd.DataFrame({"batch": ["A", "B", np.nan, "A", np.nan]})
        var = pd.DataFrame(index=["gene1", "gene2", "gene3"])
        adata = ad.AnnData(X=data, obs=obs, var=var)

        # Apply the function to coerce NaNs to 'NA'
        adata_corrected = coerce_nans_to_batch(adata, batch="batch")
        print(adata_corrected.obs["batch"].unique())
        # ['A' 'B' 'NA']

    """
    adata = adata.copy()

    if adata.obs[batch].isna().sum() > 0:
        logger.info(f" coerce_nans_to_batch: Replacing NaNs in {batch} with 'NA'...")
        adata.obs[batch] = adata.obs[batch].fillna("NA")
    else:
        logger.info(f" No NaNs found in {batch}.")

    return adata


def drop_singleton_batches(
    adata: ad.AnnData,
    batch: str,
) -> ad.AnnData:
    """Drop samples from batches that contain only a single sample.

    Parameters
    ----------
    adata
        Annotated data matrix, where rows are cells and columns are features
    batch
        Name of the batch feature in obs, the variation associated with this feature will be corrected

    Returns
    -------
    Annotated data matrix with samples from singleton batches removed

    Examples
    --------
    .. code-block:: python

        import anndata as ad
        import pandas as pd
        import numpy as np
        from alphapepttools.pp.batch_correction import drop_singleton_batches

        # Create a sample AnnData object with singleton batches
        data = np.random.rand(5, 3)
        obs = pd.DataFrame({"batch": ["A", "B", "B", "C", "D"]})
        var = pd.DataFrame(index=["gene1", "gene2", "gene3"])
        adata = ad.AnnData(X=data, obs=obs, var=var)

        # Apply the function to drop singleton batches
        adata_filtered = drop_singleton_batches(adata, batch="batch")
        print(adata_filtered.shape)  # (2, 3) - only samples from batch "B" remain

    """
    adata = adata.copy()

    # Always coerce NA values in the batch column into a single "NA" batch
    # suppress logging here:
    adata = coerce_nans_to_batch(adata, batch)

    # If any level of the to-correct data has only one level, drop it
    batch_level_count = adata.obs.groupby(batch)[batch].transform("count")

    if any(batch_level_count == 1):
        singleton_batch_samples = adata.obs.loc[batch_level_count == 1, batch]
        logger.info(
            f" coerce_nans_to_batch: Some levels of batch '{batch}' have only one sample. Dropped samples and their respective value in '{batch}': {list(zip(singleton_batch_samples.index.tolist(), singleton_batch_samples.tolist(), strict=False))}"
        )
        return adata[batch_level_count > 1, :].copy()
    logger.info(f" coerce_nans_to_batch: No singleton batches found for {batch}.")

    return adata


def scanpy_pycombat(
    adata: ad.AnnData, batch: str, covariates: None | list[str] = None, layer: str | None = None, *, copy: bool = False
) -> ad.AnnData:
    """Correct batch effects using the ComBat method :cite:`Johnson.2007`.

    Applies empirical Bayes batch correction to remove systematic
    non-biological variation associated with a batch variable in
    ``adata.obs``. The input data must be free of NaN values. Remove features with missing values or run an appropriate imputation method before calling this function.

    Uses :func:`scanpy.pp.combat` under the hood (:cite:`Wolf.2018`).

    Parameters
    ----------
    adata
        Annotated data matrix, where rows are cells and columns are features. The data matrix
        cannot contain NaN values
    batch
        Name of the batch feature in obs, the variation associated with this feature will be corrected.
        Missing values in this column will be replaced by one single "NA" batch
    covariates
        Columns in `adata.obs` indicating real biological variation. Variation associated with these columns will be retained.
    layer
        Name of the layer to batch correct. If None (default), the attribute adata.X is used
    copy
        Whether to return a modified copy (True) of the anndata object. If False (default)
        modifies the object inplace

    Returns
    -------
    AnnData with batch correction applied to layer.
    If `copy=False` modifies the anndata object at layer inplace and returns None. If `copy=True`,
    returns a modified copy

    Examples
    --------
    .. code-block:: python

        import anndata as ad
        import pandas as pd
        import numpy as np
        import alphapepttools as at

        # Create example data with batch effects
        np.random.seed(0)
        batch1_data = np.random.randn(3, 4) + 1  # Batch 1 with offset
        batch2_data = np.random.randn(3, 4) - 1  # Batch 2 with offset
        X = np.vstack([batch1_data, batch2_data])

        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame({"batch": ["B1", "B1", "B1", "B2", "B2", "B2"]}),
            var=pd.DataFrame(index=["protein1", "protein2", "protein3", "protein4"]),
        )

        # Apply batch correction
        at.pp.scanpy_pycombat(adata, batch="batch")
        print("Batch correction applied to adata.X")

    References
    ----------
    .. [1] Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics (Oxford, England), 8(1), 118-127. https://doi.org/10.1093/biostatistics/kxj037
    .. [2] Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY: large-scale single-cell gene expression data analysis. Genome biology, 19(1), 15. https://doi.org/10.1186/s13059-017-1382-0
    """
    adata = adata.copy() if copy else adata
    logger.info(f" scanpy_pycombat: pply pyComBat to correct for {batch}")

    # Ensure that X is numeric
    adata.X = adata.X.astype(float)

    # Harmonize NA values in the batch column into a single "NA" batch
    if any(adata.obs[batch].isna()):
        logger.info(f" scanpy_pycombat: Found NaNs in {batch}, coercing them into a 'NA' batch...")
        adata = coerce_nans_to_batch(adata, batch)

    # NaN check: if any nans are present in the data, batch correction cannot be performed
    if any(np.isnan(adata.X).sum(axis=1) > 0):
        logger.warning(
            " scanpy_pycombat: Error in batch correction: Data matrix contains NaN values. This causes scanpy.pp.combat to fail and return all nans, therefore batch correction can not be performed. Consider running 'impute_gaussian' or another imputation prior to batch correction to remove these samples."
        )
        raise ValueError(" scanpy_pycombat: Data matrix contains NaN values.")

    # Batch correct; apparently pyCombat from scanpy sets everything to nan if there is a batch that contains only one cell
    # i.e. if a sample only occurs once per plate, everything fails and we get all nans
    # this is a known issue, but it is not clear how to fix it https://github.com/scverse/scanpy/issues/1175
    # for now, we will just catch the error and report it.
    if any(adata.obs[batch].value_counts() == 1):
        logger.warning(
            f" scanpy_pycombat: Error in batch correction: At least one batch of {batch} contains only one single sample. This causes scanpy.pp.combat to fail and return all nans, therefore batch correction can not be performed. Consider running 'drop_singleton_batches' prior to batch correction to remove these samples."
        )
        raise ValueError(" scanpy_pycombat: At least one batch contains only one single sample.")

    # Adjust to scanpy API
    batch_corrected_result = scanpy.pp.combat(adata, key=batch, covariates=covariates, inplace=False)

    if layer is None:
        adata.X = batch_corrected_result
    else:
        adata.layers[layer] = batch_corrected_result

    return adata if copy else None
