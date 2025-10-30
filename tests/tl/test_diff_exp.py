from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind

from alphatools import tl
from alphatools.pp import nanlog
from alphatools.tl.defaults import tl_defaults
from alphatools.tl.diff_exp.alphaquant import _standardize_alphaquant_results
from alphatools.tl.diff_exp.ttest import _standardize_diff_exp_ttest_results


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
    result = tl.nan_safe_ttest_ind(a, b, min_valid_values=min_valid_values)

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

    results = tl.diff_exp_ttest(
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
    fdrs = tl.nan_safe_bh_correction(pvalues)

    # Build expected dataframe for comparison with standardized columns
    comparison_key = f"{comparison[0]}_VS_{comparison[1]}"
    expected_df = pd.DataFrame(
        {
            "condition_pair": [comparison_key] * len(deltas),
            "protein": example_data.columns.tolist(),
            "log2fc": deltas,
            "p_value": pvalues,
            "-log10(p_value)": [-np.log10(p) if p != 0 and not np.isnan(p) else np.nan for p in pvalues],
            "fdr": fdrs,
            "-log10(fdr)": [-np.log10(f) if f != 0 and not np.isnan(f) else np.nan for f in fdrs],
            "method": ["ttest"] * len(deltas),
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


# Test diff_exp_alphaquant by loading small example datasets
# TODO: Mock alphaquant itself to avoid dependency on it for testing diff_exp_alphaquant; this might depend on implementing an AlphaQuant API to avoid the temp-file construct currently required
def test_diff_exp_alphaquant():
    """Testing function to ascertain stable functionality of diff_exp_alphaquant on small example datasets.

    The expected data were generated in alphatools/tests/tl/tl_test_data.ipynb and saved
    as .pkl files in alphatools/tests/tl/tl_test_data.

    """

    test_data_dir = Path(__file__).parent / "tl_test_data"
    report = pd.read_csv(test_data_dir / "example_dataset_mouse_sn_top20peptides.tsv", sep="\t")
    samplemap = pd.read_csv(test_data_dir / "samplemap_200.tsv", sep="\t")

    adata = ad.AnnData(
        X=pd.DataFrame(np.zeros(samplemap.shape[0]), index=samplemap["sample"], columns=["dummy"]),
        obs=samplemap.set_index("sample"),
    )

    comparison_key, results = tl.diff_exp_alphaquant(
        adata=adata,
        report=report,
        between_column="condition",
        comparison=("brain", "kidney"),
        min_valid_values=2,
        valid_values_filter_mode="either",
        plots="hide",
    )

    # Load expected results
    expected_comparison_key = "brain_VS_kidney"
    expected_results = {
        "protein": pd.read_pickle(test_data_dir / "alphaquant_protein_diffexp.pkl"),
        "proteoform": pd.read_pickle(test_data_dir / "alphaquant_proteoform_diffexp.pkl"),
        "peptide": pd.read_pickle(test_data_dir / "alphaquant_peptide_diffexp.pkl"),
    }

    assert comparison_key == expected_comparison_key, (
        f"Expected comparison key {expected_comparison_key}, got {comparison_key}"
    )
    for level in ["protein", "proteoform", "peptide"]:
        pd.testing.assert_frame_equal(
            results[level],
            expected_results[level],
        )


@pytest.fixture
def example_adata_ebayes():
    """AnnData fixture with example data and metadata for eBayes tests."""
    X = pd.DataFrame(
        {
            "X1": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
            "X2": [1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
            "X3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "X4": [1, 2, 3, 4, np.nan, 6, 7, 8, 9, np.nan],
        },
        index=[f"cell{i}" for i in range(10)],
    ).astype(float)

    obs = pd.DataFrame(
        {
            "group": ["B", "B", "B", "B", "B", "A", "A", "A", "A", "A"],
        },
        index=[f"cell{i}" for i in range(10)],
    )

    adata = ad.AnnData(X=X, obs=obs)

    # data has to be log-transformed
    nanlog(adata)

    return adata


# Test diff_exp_limma by loading small example datasets
@pytest.mark.parametrize(
    ("expected_df", "comparison", "expected_comparison_key", "between_column", "min_valid_values"),
    [
        (
            pd.DataFrame(
                {
                    "condition_pair": ["B_VS_A", "B_VS_A", "B_VS_A"],
                    "protein": ["X1", "X2", "X3"],
                    "log2fc": [-0.7979892670672148, -2.8389205950315937, -1.5954559846999832],
                    "p_value": [0.0030228412720276574, 6.564170711303982e-05, 0.002166002226930997],
                    "-log10(p_value)": [2.5195846568222966, 4.182820132979508, 2.664341101199458],
                    "fdr": [0.0030228412720276574, 0.00019692512133911947, 0.0030228412720276574],
                    "-log10(fdr)": [2.5195846568222966, 3.7056988782598457, 2.5195846568222966],
                    "method": ["limma_ebayes_inmoose", "limma_ebayes_inmoose", "limma_ebayes_inmoose"],
                    "stat": [-3.804007666004946, -6.2764817445960475, -3.9999011197249166],
                    "B": [-1.8736068846693623, 2.010219213410613, -1.5373419998901845],
                    "AveExpr": [4.175828737355294, 2.8008384166375, 2.1791061114716954],
                    "max_level_1_samples": [5, 5, 5],
                    "max_level_2_samples": [5, 5, 5],
                },
                index=["X1", "X2", "X3"],
            ),
            ("B", "A"),  # ensure that patsy's alphabetical ordering is cancelled out correctly
            "B_VS_A",
            "group",
            5,  # Ensure that the feature with insufficient valid values is dropped
        ),
    ],
)
def test_diff_exp_limma(
    example_adata_ebayes,
    comparison,
    expected_comparison_key,
    expected_df,
    between_column,
    min_valid_values,
):
    """Testing function to ascertain stable functionality of diff_exp_limma on a small example dataset."""

    adata = example_adata_ebayes.copy()

    comparison_key, results = tl.diff_exp_ebayes(
        adata=adata,
        between_column=between_column,
        comparison=comparison,
        min_valid_values=min_valid_values,
    )

    pd.testing.assert_frame_equal(
        results,
        expected_df,
    )

    assert comparison_key == expected_comparison_key, (
        f"Expected comparison key {expected_comparison_key}, got {comparison_key}"
    )


# Check standardization of the ttest output
@pytest.mark.parametrize(
    ("comparison_key", "input_df", "neg_log_pval", "neg_log_fdr"),
    [
        (
            "A_VS_B",
            pd.DataFrame(
                {
                    "delta_A_VS_B": [1],
                    "pvalue_A_VS_B": [1],
                    "padj_A_VS_B": [1],
                }
            ),
            -0.0,
            -0.0,
        ),
    ],
)
def test__standardize_diff_exp_ttest_results(comparison_key, input_df, neg_log_pval, neg_log_fdr):
    """Test that _standardize_diff_exp_ttest_results correctly parses all columns."""
    result = _standardize_diff_exp_ttest_results(comparison_key, input_df)

    # Check that columns match the standard DIFF_EXP_COLS
    assert list(result.columns) == tl_defaults.DIFF_EXP_COLS, (
        f"Expected columns {tl_defaults.DIFF_EXP_COLS}, got {list(result.columns)}"
    )

    # Check that the pvalue log transformation worked
    assert result["-log10(p_value)"].iloc[0] == neg_log_pval, (
        f"Expected -log10(p_value) {neg_log_pval}, got {result['-log10(p_value)'].iloc[0]}"
    )

    # Check that the fdr log transformation worked
    assert result["-log10(fdr)"].iloc[0] == neg_log_fdr, (
        f"Expected -log10(fdr) {neg_log_fdr}, got {result['-log10(fdr)'].iloc[0]}"
    )


# Check standardization of the AlphaQuant output
@pytest.mark.parametrize(
    ("comparison_key", "level", "input_df", "expected_columns", "neg_log10_pval", "neg_log10_fdr", "peptide"),
    [
        (
            "A_VS_B",
            "protein",
            pd.DataFrame(
                {
                    "condition_pair": ["A_VS_B"],
                    "protein": ["PROT_1"],
                    "log2fc": [1],
                    "p_value": [1],
                    "fdr": [1],
                    "quality_score": [1],
                }
            ),
            [*tl_defaults.DIFF_EXP_COLS, "quality_score"],
            -0.0,
            -0.0,
            None,
        ),
        (
            "A_VS_B",
            "proteoform",
            pd.DataFrame(
                {
                    "protein": ["PROT_1"],
                    "log2fc": [1],
                    "proteoform_pval": [1],
                    "proteoform_fdr": [1],
                    "proteoform_id": ["PF_1"],
                    "peptides": ["PEP1;PEP2"],
                    "num_peptides": [1],
                    "quality_score": [1],
                }
            ),
            [*tl_defaults.DIFF_EXP_COLS, "proteoform_id", "peptides", "num_peptides", "quality_score"],
            -0.0,
            -0.0,
            None,
        ),
        (
            "A_VS_B",
            "peptide",
            pd.DataFrame(
                {
                    "condition_pair": ["A_VS_B"],
                    "protein": ["PROT_1"],
                    "log2fc": [1],
                    "p_value": [1],
                    "fdr": [1],
                    "sequence": ["SEQ_PEPTIDE_"],
                    "quality_score": [1],
                }
            ),
            [*tl_defaults.DIFF_EXP_COLS, "sequence", "quality_score"],
            -0.0,
            -0.0,
            "PEPTIDE",
        ),
    ],
)
def test__standardize_alphaquant_results(
    comparison_key, level, input_df, expected_columns, neg_log10_pval, neg_log10_fdr, peptide
):
    """Test that _standardize_alphaquant_results correctly parses all columns for each level."""

    result = _standardize_alphaquant_results(comparison_key, level, input_df)

    # Check that columns match the expected columns for the level
    if level in {"protein", "proteoform"}:
        assert list(result.columns) == expected_columns, (
            f"Expected columns {expected_columns}, got {list(result.columns)}"
        )
    elif level == "peptide":
        assert list(result.columns) == expected_columns, (
            f"Expected columns {expected_columns}, got {list(result.columns)}"
        )
        # Check that peptide names were cleaned correctly
        assert result["sequence"].iloc[0] == peptide, (
            f"Expected cleaned peptide name {peptide}, got {result['sequence'].iloc[0]}"
        )

    # Check that the pvalue log transformation worked
    assert result["-log10(p_value)"].iloc[0] == neg_log10_pval, (
        f"Expected -log10(p_value) {neg_log10_pval}, got {result['-log10(p_value)'].iloc[0]}"
    )

    # Check that the fdr log transformation worked
    assert result["-log10(fdr)"].iloc[0] == neg_log10_fdr, (
        f"Expected -log10(fdr) {neg_log10_fdr}, got {result['-log10(fdr)'].iloc[0]}"
    )
