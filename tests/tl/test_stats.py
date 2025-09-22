import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control, ttest_ind

from alphatools.tl.stats import nan_safe_bh_correction, nan_safe_ttest_ind


@pytest.mark.parametrize(
    "p_values",
    [
        # Test mixed case with NaNs interspersed
        np.array([0.01, 0.05, np.nan, 0.001, np.nan]),
        # Test all NaN case
        np.array([np.nan, np.nan, np.nan]),
        # Test normal case without NaNs
        np.array([0.01, 0.05, 0.001, 0.1]),
        # Test single value case
        np.array([0.05]),
        # Test single NaN case
        np.array([np.nan]),
        # Test empty array case
        np.array([]),
    ],
)
def test_nan_safe_bh_correction(p_values):
    """Test that nan_safe_bh_correction preserves order and handles NaNs correctly."""
    corrected_pvals = nan_safe_bh_correction(p_values)

    na_positions = np.isnan(p_values)
    valid_pvalues = p_values[~na_positions]
    expected = false_discovery_control(valid_pvalues, method="bh") if np.any(~na_positions) else np.array([])

    # Bit verbose but stitching back together expected with NaNs in original positions
    expected_full = []
    for val in p_values:
        if np.isnan(val):
            expected_full.append(np.nan)
        else:
            expected_full.append(expected[0])
            # Get rid of first element to get the next with [0]
            expected = expected[1:]

    expected = np.array(expected_full)

    # Use allclose with equal_nan=True for comparison
    assert np.allclose(corrected_pvals, expected, equal_nan=True), f"Expected {expected}, got {corrected_pvals}"
    assert np.array_equal(np.isnan(corrected_pvals), np.isnan(p_values)), "NaN positions changed"


@pytest.mark.parametrize(
    ("ab", "expected"),
    [
        # Both series with sufficient data
        ((pd.Series([1, 2, 3]), pd.Series([4, 5, 6])), "ttest"),
        # One series with insufficient data
        ((pd.Series([1]), pd.Series([4, 5, 6])), (np.nan, np.nan)),
        # Both series with insufficient data
        ((pd.Series([np.nan]), pd.Series([np.nan])), (np.nan, np.nan)),
        # Series with NaNs but sufficient data
        ((pd.Series([1, 2, np.nan]), pd.Series([4, 5, np.nan])), "ttest"),
        # Empty series
        ((pd.Series([]), pd.Series([])), (np.nan, np.nan)),
    ],
)
def test_nan_safe_ttest_ind(ab, expected):
    """Test that nan_safe_ttest_ind handles NaNs and insufficient data correctly."""
    a, b = ab
    result = nan_safe_ttest_ind(a, b)

    if expected == "ttest":
        expected_result = ttest_ind(a.dropna(), b.dropna())
        assert np.allclose(result, expected_result), f"Expected {expected_result}, got {result}"
    else:
        assert result == expected, f"Expected {expected}, got {result}"
