"""Helpers that fill the columns shared by all differential expression result tables."""

import anndata as ad
import numpy as np

from alphapepttools.tl.defaults import tl_defaults


def negative_log10_pvalue(pvalue: float, ceiling: float = tl_defaults.CEILING_NEGATIVE_LOG10_PVALUE) -> float:
    """Convert p-value to -log10 scale with special handling for zero values.

    Parameters
    ----------
    pvalue
        P-value to convert.
    ceiling
        Value to use when pvalue is exactly 0. Default is 300.

    Returns
    -------
    float
        -log10(pvalue), or ceiling if pvalue is exactly 0.

    Examples
    --------
    .. code-block:: python

        from alphapepttools.tl.diff_exp._results import negative_log10_pvalue

        # Regular p-value
        result = negative_log10_pvalue(0.01)
        # Returns 2.0

        # Zero p-value returns ceiling value
        result = negative_log10_pvalue(0.0)
        # Returns 300.0

    """
    return ceiling if pvalue == 0 else -np.log10(pvalue)


def determine_max_replicates(
    adata: ad.AnnData,
    between_column: str,
    level_1: str,
    level_2: str,
) -> tuple[int, int]:
    """Determine maximum number of replicates for each level and log the counts.

    Parameters
    ----------
    adata
        The AnnData object containing the data.
    between_column
        The column name in adata.obs to use for grouping.
    level_1
        First group/level name.
    level_2
        Second group/level name.

    Returns
    -------
    tuple[int, int]
        Number of samples for (level_1, level_2).

    Examples
    --------
    .. code-block:: python

        import anndata as ad
        import numpy as np
        from alphapepttools.tl.diff_exp._results import determine_max_replicates

        # Create example data with two groups
        adata = ad.AnnData(X=np.random.rand(6, 5), obs={"condition": ["A", "A", "A", "B", "B", "B"]})

        counts = determine_max_replicates(adata, "condition", "A", "B")
        # Returns (3, 3)

    """
    max_samples_level_1 = adata.obs[adata.obs[between_column] == level_1].shape[0]
    max_samples_level_2 = adata.obs[adata.obs[between_column] == level_2].shape[0]
    return max_samples_level_1, max_samples_level_2
