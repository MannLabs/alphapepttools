import contextlib
import logging
from collections.abc import Iterable

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from alphapepttools._matrix import get_obs
from alphapepttools.tl.defaults import tl_defaults

logger = logging.getLogger(__name__)


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

        from alphapepttools.tl.utils import negative_log10_pvalue

        # Regular p-value
        result = negative_log10_pvalue(0.01)
        # Returns 2.0

        # Zero p-value returns ceiling value
        result = negative_log10_pvalue(0.0)
        # Returns 300.0

    """
    return ceiling if pvalue == 0 else -np.log10(pvalue)


@contextlib.contextmanager
def _suppress_plots():  # noqa: ANN202 # avoid generator return type annotation
    """Context manager to suppress matplotlib plot display."""
    original_show = plt.show
    plt.show = lambda *a, **k: None  # NOQA: ARG005  # ty: ignore[invalid-assignment]  # intentional monkeypatch to suppress display
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
        from alphapepttools.tl.utils import determine_max_replicates

        # Create example data with two groups
        adata = ad.AnnData(X=np.random.rand(6, 5), obs={"condition": ["A", "A", "A", "B", "B", "B"]})

        counts = determine_max_replicates(adata, "condition", "A", "B")
        # Returns (3, 3)

    """
    obs = get_obs(adata)
    max_samples_level_1 = obs[obs[between_column] == level_1].shape[0]
    max_samples_level_2 = obs[obs[between_column] == level_2].shape[0]
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
    adata
        The AnnData object containing the data.
    between_column
        The column name in adata.obs to use for grouping.
    comparison
        The two groups to compare.
    min_valid_values
        The minimum number of valid (non-NaN) values required.

    Returns
    -------
    ad.AnnData
        A copy of the filtered AnnData object.

    Examples
    --------
    .. code-block:: python

        import anndata as ad
        import numpy as np
        from alphapepttools.tl.utils import drop_features_with_too_few_valid_values

        # Create data with some missing values
        X = np.array(
            [
                [1, 2, np.nan, 4],
                [5, np.nan, np.nan, 8],
                [9, 10, 11, 12],
                [np.nan, 14, 15, 16],
                [17, 18, np.nan, 20],
                [21, 22, 23, 24],
            ]
        )

        adata = ad.AnnData(
            X=X, obs={"group": ["A", "A", "A", "B", "B", "B"]}, var={"feature": ["F1", "F2", "F3", "F4"]}
        )

        # Keep only features with at least 2 valid values per group
        filtered = drop_features_with_too_few_valid_values(adata, "group", ("A", "B"), min_valid_values=2)

        # in this case, features 1, 2 and 4 would be kept, while feature 3 would be dropped due to too many NaNs

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

    This function checks:
    - The between_column exists in adata.obs
    - The comparison is a tuple of exactly two elements
    - Both groups in the comparison exist in the data
    - Each group has at least min_valid_values samples

    Parameters
    ----------
    adata
        AnnData object with features and observations.
    between_column
        Name of the column in adata.obs that contains the groups to compare.
    comparison
        Tuple of exactly two group names to compare (group1, group2).
    min_valid_values
        Minimum number of samples required per group.

    Returns
    -------
    tuple[str, str]
        Validated group names (g1, g2).

    Raises
    ------
    ValueError
        If any validation check fails.

    Examples
    --------
    .. code-block:: python

        import anndata as ad
        import numpy as np
        from alphapepttools.tl.utils import validate_ttest_inputs

        # Create example data
        adata = ad.AnnData(X=np.random.rand(6, 5), obs={"condition": ["A", "A", "A", "B", "B", "B"]})

        # Validate comparison between two groups
        g1, g2 = validate_ttest_inputs(adata, "condition", ("A", "B"), min_valid_values=2)
        # Returns ('A', 'B')

    """
    # check that between column exists in obs
    if between_column not in adata.obs.columns:
        raise ValueError(f"Error in group_ratios_ttest_ind: Column '{between_column}' not found in adata.obs.")

    # validate comparison tuple
    if not isinstance(comparison, tuple) or len(comparison) != 2:  # noqa: PLR2004 since this is a tuple check
        raise ValueError("Error in group_ratios_ttest_ind: 'comparison' must be a tuple of exactly two group names.")

    g1, g2 = comparison
    available_groups = set(get_obs(adata)[between_column].dropna().unique())

    # check that both groups exist in the data
    if g1 not in available_groups:
        raise ValueError(f"Error in group_ratios_ttest_ind: Group '{g1}' not found in column '{between_column}'.")
    if g2 not in available_groups:
        raise ValueError(f"Error in group_ratios_ttest_ind: Group '{g2}' not found in column '{between_column}'.")

    # check that each group has sufficient samples to even meet min_valid_values
    group_counts = get_obs(adata)[between_column].value_counts()
    if group_counts[g1] < min_valid_values:
        raise ValueError(
            f"Error in group_ratios_ttest_ind: Group '{g1}' has only {group_counts[g1]} samples, need at least {min_valid_values}."
        )
    if group_counts[g2] < min_valid_values:
        raise ValueError(
            f"Error in group_ratios_ttest_ind: Group '{g2}' has only {group_counts[g2]} samples, need at least {min_valid_values}."
        )

    return g1, g2


def find_iterable_kwargs(
    kwargs_dict: dict,
    match_length: int | None = None,
) -> dict:
    """Find kwargs whose values are array-like (not strings/bytes/dicts).

    Parameters
    ----------
    kwargs_dict
        A dictionary of keyword arguments to check.

    Returns
    -------
        A dictionary containing only those key-value pairs from kwargs_dict where the value is an iterable and, if match_length is specified, has the specified length.

    """
    if not isinstance(kwargs_dict, dict):
        raise TypeError(f"Input must be a dictionary of keyword arguments, got {type(kwargs_dict).__name__}.")

    return {
        k: v
        for k, v in kwargs_dict.items()
        if isinstance(v, Iterable)
        and not isinstance(v, (str, bytes, dict))
        and (match_length is None or len(v) == match_length)
    }
