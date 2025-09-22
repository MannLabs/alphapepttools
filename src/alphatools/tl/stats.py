import logging

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ttest_ind

# logging configuration
logging.basicConfig(level=logging.INFO)


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
    # convert array to dataframe with distinct index
    pval_df = pd.DataFrame({"pvals": pvals})

    initial_index = range(len(pval_df))
    pval_df.index = initial_index

    pval_df_no_nans = pval_df.copy().dropna()
    pval_df_no_nans["pvals_corrected"] = false_discovery_control(pval_df_no_nans["pvals"], method="bh")

    # merge back to original index
    pval_df = pval_df.join(pval_df_no_nans["pvals_corrected"], how="left")

    # verify that the original index is preserved
    if not all(pval_df.index == initial_index):
        raise ValueError("Index mismatch in nan_safe_bh_correction.")

    return pval_df["pvals_corrected"].to_numpy()


def nan_safe_ttest_ind(
    a: pd.Series,
    b: pd.Series,
    **kwargs,
) -> tuple:
    """NaN-safe wrapper around scipy.stats.ttest_ind.

    Performs independent t-test between two samples, but returns (nan, nan) if either
    input has fewer than two non-NaN values. Automatically converts inputs to pandas
    Series if needed.

    Parameters
    ----------
    a : pd.Series
        First sample for comparison.
    b : pd.Series
        Second sample for comparison.
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
    MIN_VALS = 2

    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        try:
            a = pd.Series(a)
            b = pd.Series(b)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Cannot convert inputs to pandas Series: {e}") from e

    if a.count() < MIN_VALS or b.count() < MIN_VALS:
        return (np.nan, np.nan)

    if "nan_policy" not in kwargs:
        kwargs["nan_policy"] = "omit"

    return ttest_ind(a, b, **kwargs)
