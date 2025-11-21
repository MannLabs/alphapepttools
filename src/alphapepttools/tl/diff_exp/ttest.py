import logging

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from alphapepttools.pp.data import filter_by_metadata
from alphapepttools.tl.defaults import tl_defaults
from alphapepttools.tl.stats import nan_safe_bh_correction
from alphapepttools.tl.utils import determine_max_replicates, negative_log10_pvalue, validate_ttest_inputs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    >>> from alphapepttools.tl.stats import nan_safe_ttest_ind
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


def _standardize_diff_exp_ttest_results(
    comparison_key: str,
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize ttest results DataFrames

    To harmonize the output of the standard ttest with other methods,
    we rename some columns and add columns for context.

    Parameters
    ----------
    comparison_key : str
        Identifier for the comparison, e.g. "group1_VS_group2".
    result_df : pd.DataFrame
        DataFrame with ttest results.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns outlined in the common DIFF_EXP_COLS.

    """
    result_df = result_df.copy()

    # Map columns from ttest output format to standard format
    current_column_map = {
        f"delta_{comparison_key}": "log2fc",
        f"pvalue_{comparison_key}": "p_value",
        f"padj_{comparison_key}": "fdr",
    }
    result_df = result_df.rename(columns=current_column_map)

    # Add standard columns
    result_df["condition_pair"] = comparison_key
    result_df["protein"] = result_df.index
    result_df["method"] = "ttest"

    # For p-values of exactly 0, use a very large value instead of NaN
    result_df["-log10(p_value)"] = result_df["p_value"].apply(negative_log10_pvalue)
    result_df["-log10(fdr)"] = result_df["fdr"].apply(negative_log10_pvalue)

    # Reorder columns to match DIFF_EXP_COLS
    return result_df[tl_defaults.DIFF_EXP_COLS].copy()


def diff_exp_ttest(
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
    g1, g2 = validate_ttest_inputs(adata, between_column, comparison, min_valid_values)
    print(f"Comparing groups: {g1} vs {g2}")

    # perform single comparison between the two specified groups
    g1_df = filter_by_metadata(adata, {between_column: g1}, axis=0).to_df()
    g2_df = filter_by_metadata(adata, {between_column: g2}, axis=0).to_df()

    comparison_name = f"{g1}_VS_{g2}"
    features = pd.Series(adata.var_names)

    # Record maximum number of replicates per group
    max_samples_g1, max_samples_g2 = determine_max_replicates(adata, between_column, g1, g2)

    # record number of non-na samples in each group
    g1_n_samples = g1_df.count()
    g2_n_samples = g2_df.count()

    # calculate ratio and difference, latter for log-transformed inputs
    # Handle the intricacy that there can be a delta even when the ratio would be a division by zero.
    # First calculate means and the delta, then replace zero means with NaN for ratio calculation
    g1_mean = g1_df.mean(axis=0)
    g2_mean = g2_df.mean(axis=0)
    delta = g1_mean - g2_mean

    # where mean is zero, insert np.nan to avoid division by zero in the ratios
    g1_mean[g1_mean == 0] = np.nan
    g2_mean[g2_mean == 0] = np.nan
    ratio = g1_mean / g2_mean

    # Perform t-test for each feature between the two groups
    # Lambda captures DataFrames at creation time to avoid closure issues
    t_values, p_values = zip(
        *features.apply(
            lambda x, _g1_df=g1_df, _g2_df=g2_df: nan_safe_ttest_ind(
                a=_g1_df[x],
                b=_g2_df[x],
                equal_var=equal_var,
                nan_policy="omit",
                min_valid_values=min_valid_values,
            )
        ),
        strict=False,
    )

    # adjust pvalues using Benjamini-Hochberg method, accounting for nans
    p_adj = nan_safe_bh_correction(p_values)

    # store results in dataframe
    result_df = (
        pd.DataFrame(
            {
                "id": features.to_numpy(),
                f"ratio_{comparison_name}": ratio,
                f"delta_{comparison_name}": delta,
                f"tvalue_{comparison_name}": t_values,
                f"pvalue_{comparison_name}": p_values,
                f"padj_{comparison_name}": p_adj,
                f"n_{g1}": g1_n_samples,
                f"n_{g2}": g2_n_samples,
                "max_level_1_samples": max_samples_g1,
                "max_level_2_samples": max_samples_g2,
            }
        )
        .set_index("id", drop=True)
        .rename_axis(index=None)
    )

    return _standardize_diff_exp_ttest_results(comparison_name, result_df)
