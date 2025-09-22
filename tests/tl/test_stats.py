import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control, ttest_ind

from alphatools.tl.stats import group_ratios_ttest_ind, nan_safe_bh_correction, nan_safe_ttest_ind


@pytest.fixture
def example_data():
    def make_dummy_data():
        return pd.DataFrame(
            {
                "X1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # all valid values
                "X2": [10, np.nan, 12, 13, 14, 15, 16, np.nan, 18, np.nan],  # some NaNs interspersed
                "X3": [5, 6, 7, 8, 9, 10, 11, 12, np.nan, np.nan],  # 3 valid values for group B
                "X4": [5, 6, 7, 8, 9, 10, np.nan, np.nan, np.nan, np.nan],  # only one valid value for group B
                "X5": [np.nan] * 10,  # all NaNs
                "X6": [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],  # all valid values with zero mean
            },
            index=[f"cell{i}" for i in range(10)],
        )

    return make_dummy_data()


@pytest.fixture
def example_metadata():
    def make_dummy_metadata():
        return pd.DataFrame(
            {
                "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            },
            index=[f"cell{i}" for i in range(10)],
        )

    return make_dummy_metadata()


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
    ("ab", "expected", "min_valid_values"),
    [
        # Both series with sufficient data
        ((pd.Series([1, 2, 3]), pd.Series([4, 5, 6])), "ttest", None),
        # One series with insufficient data
        ((pd.Series([1]), pd.Series([4, 5, 6])), (np.nan, np.nan), None),
        # Both series with insufficient data
        ((pd.Series([np.nan]), pd.Series([np.nan])), (np.nan, np.nan), None),
        # Series with NaNs but sufficient data
        ((pd.Series([1, 2, np.nan]), pd.Series([4, 5, np.nan])), "ttest", None),
        # Empty series
        ((pd.Series([]), pd.Series([])), (np.nan, np.nan), None),
        # Series with insufficient data due to min_valid_values being higher than the default of 2
        ((pd.Series([1, 2, 3]), pd.Series([4, 5, 6, 7, 8])), (np.nan, np.nan), 4),
    ],
)
def test_nan_safe_ttest_ind(ab, expected, min_valid_values):
    """Test that nan_safe_ttest_ind handles NaNs and insufficient data correctly."""
    a, b = ab
    result = nan_safe_ttest_ind(a, b, min_valid_values=min_valid_values)

    if expected == "ttest":
        expected_result = ttest_ind(a.dropna(), b.dropna())
        assert np.allclose(result, expected_result), f"Expected {expected_result}, got {result}"
    else:
        assert result == expected, f"Expected {expected}, got {result}"


# Test group-wise ttest with ratios
@pytest.mark.parametrize(
    ("between_column", "comparison", "min_valid_values", "expected_output"),
    [
        # Standard case with two valid values filter
        (
            "group",
            ("A", "B"),
            2,  # default
            {
                "X1": "ttest",
                "X2": "ttest",
                "X3": "ttest",
                "X4": "nans",  # Only one valid in group B
                "X5": "nans",  # All NaNs
                "X6": "ttest",  # zero mean in group B but sufficient values
            },
        ),
        # Case with min_valid_values set to 4
        (
            "group",
            ("A", "B"),
            4,  # stricter filter
            {
                "X1": "ttest",
                "X2": "ttest",
                "X3": "nans",  # Only 3 valid in group B
                "X4": "nans",  # Only one valid in group B
                "X5": "nans",  # All NaNs
                "X6": "ttest",  # zero mean in group B but sufficient values
            },
        ),
    ],
)
def test_group_ratios_ttest_ind(
    example_data, example_metadata, between_column, comparison, min_valid_values, expected_output
):
    """Test group_ratios_ttest_ind with various scenarios."""

    adata = ad.AnnData(
        X=example_data,
        obs=example_metadata,
    )

    results = group_ratios_ttest_ind(
        adata=adata,
        between_column=between_column,
        comparison=comparison,
        min_valid_values=min_valid_values,
    )

    # Iterate over features and perform manual ttests and ratio calculation
    ratios = []
    deltas = []
    tvalues = []
    pvalues = []
    n_a = []
    n_b = []
    for f in adata.var_names:
        a = pd.Series(adata[adata.obs[between_column] == comparison[0], f].X.flatten()).dropna()
        b = pd.Series(adata[adata.obs[between_column] == comparison[1], f].X.flatten()).dropna()

        # Handle the intricacy that there can be a delta even when the ratio would be a division by zero.
        # First calculate means and the delta, then replace zero means with NaN for ratio calculation
        a_mean = a.mean()
        b_mean = b.mean()
        deltas.append(a_mean - b_mean)

        # zero mean guard for ratio
        a_mean = a_mean if a_mean != 0 else np.nan
        b_mean = b_mean if b_mean != 0 else np.nan
        ratios.append(a_mean / b_mean)

        # Emulate check for sufficient valid values, which can let
        # ratios pass but not ttests performed
        if len(a) < min_valid_values or len(b) < min_valid_values:
            tvalues.append(np.nan)
            pvalues.append(np.nan)
        # Perform ttests
        elif expected_output[f] == "ttest":
            t_stat, p_val = ttest_ind(a, b, nan_policy="omit", equal_var=False)
            tvalues.append(t_stat)
            pvalues.append(p_val)
        else:
            tvalues.append(np.nan)
            pvalues.append(np.nan)

        n_a.append(len(a))
        n_b.append(len(b))

    # Adjust pvalues
    pvalues = np.array(pvalues)
    fdrs = nan_safe_bh_correction(pvalues)

    # Build expected dataframe for comparison
    expected_df = pd.DataFrame(
        {
            f"ratio_{comparison[0]}_VS_{comparison[1]}": ratios,
            f"delta_{comparison[0]}_VS_{comparison[1]}": deltas,
            f"tvalue_{comparison[0]}_VS_{comparison[1]}": tvalues,
            f"pvalue_{comparison[0]}_VS_{comparison[1]}": pvalues,
            f"padj_{comparison[0]}_VS_{comparison[1]}": fdrs,
            f"n_{comparison[0]}": n_a,
            f"n_{comparison[1]}": n_b,
        },
        index=example_data.columns,
    )

    # Compare results
    pd.testing.assert_frame_equal(
        results,
        expected_df,
        check_exact=False,
        rtol=1e-5,
        atol=1e-8,
        check_names=False,
        check_dtype=False,
    )
