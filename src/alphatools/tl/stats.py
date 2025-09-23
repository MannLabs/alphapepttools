import logging

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ttest_ind

from alphatools.pp.data import filter_by_metadata

# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def nan_safe_bh_correction(
    pvals: np.array,
) -> np.array:
    """Apply Benjamini-Hochberg correction with NaN-safe handling.

    Scipy.stats.false_discovery_control is not nan-safe, we need to delete nans, apply correction, then re-insert nans.
    This method preserves nans in their original positions while applying BH correction to valid p-values.

    Parameters
    ----------
    pvals : np.array
        Array of p-values, may contain NaNs.

    Returns
    -------
    np.array
        Array with BH-corrected p-values, NaNs preserved in original positions.

    Examples
    --------
    >>> import numpy as np
    >>> from alphatools.tl.stats import nan_safe_bh_correction
    >>> pvals = np.array([0.01, 0.05, np.nan, 0.001, np.nan])
    >>> corrected = nan_safe_bh_correction(pvals)
    >>> # Returns [0.015, 0.05, nan, 0.015, nan] (approximately)
    """
    # Convert to numpy array if not already
    pvals = np.asarray(pvals)

    # Create output array filled with NaNs
    corrected_pvals = np.full_like(pvals, np.nan, dtype=np.float64)

    # Find indices of non-NaN values
    valid_mask = ~np.isnan(pvals)
    valid_indices = np.where(valid_mask)[0]

    # If there are valid p-values, apply BH correction
    if len(valid_indices) > 0:
        valid_pvals = pvals[valid_mask]
        corrected_valid = false_discovery_control(valid_pvals, method="bh")

        # Put corrected values back in their original positions
        corrected_pvals[valid_indices] = corrected_valid

    return corrected_pvals


def nan_safe_ttest_ind(
    a: pd.Series,
    b: pd.Series,
    min_valid_values: int | None = None,
    **kwargs,
) -> tuple[float, float] | tuple[np.nan, np.nan]:
    """NaN-safe wrapper around scipy.stats.ttest_ind.

    Performs independent t-test between two samples, but returns (nan, nan) if either
    input has fewer than two non-NaN values. Automatically converts inputs to pandas
    Series if needed. Defaults are set to omit NaNs and not assume equal variance (Welch's t-test),
    which can be changed by passing different arguments for "nan_policy" and "equal_var" to **kwargs.

    Parameters
    ----------
    a : pd.Series
        First sample for comparison.
    b : pd.Series
        Second sample for comparison.
    min_valid_values : int, optional
        Minimum number of non-NaN values required in either sample to perform t-test. Since this
        function has no means of imputation, this means that BOTH samples must have at least this many
        non-NaN values to perform the t-test. Default is 2.
    **kwargs
        Additional keyword arguments passed to scipy.stats.ttest_ind.

    Returns
    -------
    tuple
        (t_statistic, p_value) if both samples have at least 2 non-NaN values,
        otherwise (nan, nan).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from alphatools.tl.stats import nan_safe_ttest_ind
    >>> a = pd.Series([1, 2, 3, np.nan])
    >>> b = pd.Series([4, 5, 6, 7])
    >>> t_stat, p_val = nan_safe_ttest_ind(a, b)
    >>> # Returns valid t-test results since both have >= 2 non-NaN values

    >>> c = pd.Series([1, np.nan])  # Only 1 non-NaN value
    >>> t_stat, p_val = nan_safe_ttest_ind(c, b)
    >>> # Returns (nan, nan) since c has < 2 non-NaN values
    """
    # fewer than two values per group is not a valid t-test
    min_valid_values = min_valid_values or 2

    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        try:
            a = pd.Series(a)
            b = pd.Series(b)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Cannot convert inputs to pandas Series: {e}") from e

    if a.count() < min_valid_values or b.count() < min_valid_values:
        return (np.nan, np.nan)

    if "nan_policy" not in kwargs:
        kwargs["nan_policy"] = "omit"

    if "equal_var" not in kwargs:
        kwargs["equal_var"] = False

    return ttest_ind(a, b, **kwargs)


def _validate_ttest_inputs(
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


def group_ratios_ttest_ind(
    adata: ad.AnnData,
    between_column: str,
    comparison: tuple,
    min_valid_values: int = 2,
    *,
    equal_var: bool = False,
) -> pd.DataFrame | None:
    """Calculate ratios of features between two specific groups using t-test.

    Calculate ratios and log2 ratios of each feature in the AnnData object's X
    between two specific groups defined in the comparison tuple.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with features and observations.
    between_column : str
        Name of the column in adata.obs that contains the groups to compare.
    comparison : tuple
        Tuple of exactly two group names to compare (group1, group2).
    min_valid_values : int, optional
        Minimum number of samples required per group. By default 2.
    equal_var : bool, optional
        Whether to assume equal variance in the t-test. By default False.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with ratios, deltas, t-statistics, p-values, and adjusted p-values
        for the comparison between the two specified groups. Returns None if validation fails.
    """
    # Validate inputs
    g1, g2 = _validate_ttest_inputs(adata, between_column, comparison, min_valid_values)

    # perform single comparison between the two specified groups
    g1_frame = filter_by_metadata(adata, {between_column: g1}, axis=0).to_df()
    g2_frame = filter_by_metadata(adata, {between_column: g2}, axis=0).to_df()

    comparison_name = f"{g1}_VS_{g2}"
    features = pd.Series(adata.var_names)

    # record number of non-na samples in each group
    g1_n_samples = g1_frame.count()
    g2_n_samples = g2_frame.count()

    # calculate ratio and difference, latter for log-transformed inputs
    # Handle the intricacy that there can be a delta even when the ratio would be a division by zero.
    # First calculate means and the delta, then replace zero means with NaN for ratio calculation
    g1_mean = g1_frame.mean(axis=0)
    g2_mean = g2_frame.mean(axis=0)
    delta = g1_mean - g2_mean

    # where mean is zero, insert np.nan to avoid division by zero in the ratios
    g1_mean[g1_mean == 0] = np.nan
    g2_mean[g2_mean == 0] = np.nan
    ratio = g1_mean / g2_mean

    # Perform t-test for each feature between the two groups
    # Lambda captures DataFrames at creation time to avoid closure issues
    t, p = zip(
        *features.apply(
            lambda x, _g1_frame=g1_frame, _g2_frame=g2_frame: nan_safe_ttest_ind(
                a=_g1_frame[x],
                b=_g2_frame[x],
                equal_var=equal_var,
                nan_policy="omit",
                min_valid_values=min_valid_values,
            )
        ),
        strict=False,
    )

    # adjust pvalues using Benjamini-Hochberg method, accounting for nans
    p_adj = nan_safe_bh_correction(p)

    # store results in dataframe
    result_df = (
        pd.DataFrame(
            {
                "id": features.to_numpy(),
                f"ratio_{comparison_name}": ratio,
                f"delta_{comparison_name}": delta,
                f"tvalue_{comparison_name}": t,
                f"pvalue_{comparison_name}": p,
                f"padj_{comparison_name}": p_adj,
                f"n_{g1}": g1_n_samples,
                f"n_{g2}": g2_n_samples,
            }
        )
        .set_index("id", drop=True)
        .rename_axis(index=None)
    )

    return result_df  # noqa: RET504 output standardization function will go here
