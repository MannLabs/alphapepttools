import contextlib
import logging

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from alphapepttools.tl.defaults import tl_defaults

logger = logging.getLogger(__name__)


def negative_log10_pvalue(pvalue: float, ceiling: float = tl_defaults.CEILING_NEGATIVE_LOG10_PVALUE) -> float:
    """Convert p-value to -log10 scale with special handling for zero values.

    Parameters
    ----------
    pvalue : float
        P-value to convert.
    ceiling : float, optional
        Value to use when pvalue is exactly 0. Default is 300.

    Returns
    -------
    float
        -log10(pvalue), or ceiling if pvalue is exactly 0.

    """
    return ceiling if pvalue == 0 else -np.log10(pvalue)


# Context manager to suppress plots if needed
@contextlib.contextmanager
def _suppress_plots():  # noqa: ANN202 # avoid generator return type annotation
    original_show = plt.show
    plt.show = lambda *a, **k: None  # NOQA: ARG005
    try:
        yield
        plt.close("all")
    finally:
        plt.show = original_show


def determine_max_replicates(
    adata: ad.AnnData,
    between_column: str,
    level_1: str,
    level_2: str,
) -> tuple[int, int]:
    """Determine maximum number of replicates for each level and log the counts.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing the data.
    between_column : str
        The column name in adata.obs to use for grouping.
    level_1 : str
        First group/level name.
    level_2 : str
        Second group/level name.

    Returns
    -------
    tuple[int, int]
        Number of samples for (level_1, level_2).

    """
    max_samples_level_1 = adata.obs[adata.obs[between_column] == level_1].shape[0]
    max_samples_level_2 = adata.obs[adata.obs[between_column] == level_2].shape[0]
    logger.info(f"Number of samples for {level_1}: {max_samples_level_1}")
    logger.info(f"Number of samples for {level_2}: {max_samples_level_2}")
    return max_samples_level_1, max_samples_level_2


def drop_features_with_too_few_valid_values(
    adata: ad.AnnData,
    between_column: str,
    comparison: tuple[str, str],
    min_valid_values: int,
) -> ad.AnnData:
    """Drop features with too few valid (non-NaN) values in either group.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing the data.
    between_column : str
        The column name in adata.obs to use for grouping.
    comparison : tuple[str, str]
        The two groups to compare.
    min_valid_values : int
        The minimum number of valid (non-NaN) values required.

    Returns
    -------
    ad.AnnData
        A copy of the filtered AnnData object.

    """
    for lvl in comparison:
        valid_counts = adata[adata.obs[between_column] == lvl].to_df().astype(float).count(axis=0)
        features_to_keep = valid_counts >= min_valid_values
        adata = adata[:, features_to_keep].copy()

    return adata


def validate_ttest_inputs(
    adata: ad.AnnData,
    between_column: str,
    comparison: tuple,
    min_valid_values: int,
) -> tuple[str, str]:
    """Validate inputs for group_ratios_ttest_ind.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with features and observations.
    between_column : str
        Name of the column in adata.obs that contains the groups to compare.
    comparison : tuple
        Tuple of exactly two group names to compare (group1, group2).
    min_valid_values : int
        Minimum number of samples required per group.

    Returns
    -------
    tuple[str, str]
        Validated group names (g1, g2).

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    # check that between column exists in obs
    if between_column not in adata.obs.columns:
        raise ValueError(f"Error in group_ratios_ttest_ind: Column '{between_column}' not found in adata.obs.")

    # validate comparison tuple
    if not isinstance(comparison, tuple) or len(comparison) != 2:  # noqa: PLR2004 since this is a tuple check
        raise ValueError("Error in group_ratios_ttest_ind: 'comparison' must be a tuple of exactly two group names.")

    g1, g2 = comparison
    available_groups = set(adata.obs[between_column].dropna().unique())

    # check that both groups exist in the data
    if g1 not in available_groups:
        raise ValueError(f"Error in group_ratios_ttest_ind: Group '{g1}' not found in column '{between_column}'.")
    if g2 not in available_groups:
        raise ValueError(f"Error in group_ratios_ttest_ind: Group '{g2}' not found in column '{between_column}'.")

    # check that each group has sufficient samples to even meet min_valid_values
    group_counts = adata.obs[between_column].value_counts()
    if group_counts[g1] < min_valid_values:
        raise ValueError(
            f"Error in group_ratios_ttest_ind: Group '{g1}' has only {group_counts[g1]} samples, need at least {min_valid_values}."
        )
    if group_counts[g2] < min_valid_values:
        raise ValueError(
            f"Error in group_ratios_ttest_ind: Group '{g2}' has only {group_counts[g2]} samples, need at least {min_valid_values}."
        )

    return g1, g2
