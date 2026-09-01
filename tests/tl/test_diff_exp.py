from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind

from alphapepttools import tl
from alphapepttools.pp import nanlog
from alphapepttools.pp.data import filter_data_completeness
from alphapepttools.tl.defaults import tl_defaults
from alphapepttools.tl.diff_exp.alphaquant_wrapper import _HAS_ALPHAQUANT, _standardize_alphaquant_results
from alphapepttools.tl.diff_exp.ebayes import _HAS_INMOOSE
from alphapepttools.tl.diff_exp.ebayes_expanded import (
    _METHOD_NAME,
    _build_design_matrix,
    _contrasts_from_matrix,
    _make_contrasts,
    _nan_lmfit,
    _replicate_gate_mask,
    _resolve_comparison,
    _run_contrasts,
    _standardize_contrast_frame,
)
from alphapepttools.tl.diff_exp.ebayes_expanded import diff_exp_ebayes as diff_exp_ebayes_expanded
from alphapepttools.tl.diff_exp.ttest import _standardize_diff_exp_ttest_results


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


def test_nan_safe_ttest_ind_converts_non_series():
    """List inputs should be converted to pd.Series and processed normally."""
    result = tl.nan_safe_ttest_ind([1, 2, 3], [4, 5, 6])
    expected = ttest_ind([1, 2, 3], [4, 5, 6])
    assert np.allclose(result, expected)


def test_nan_safe_ttest_ind_raises_on_unconvertible_input():
    """Input that pd.Series cannot accept (e.g. a 2D array) should raise TypeError."""
    with pytest.raises(TypeError, match="Cannot convert inputs"):
        tl.nan_safe_ttest_ind(np.array([[1, 2], [3, 4]]), pd.Series([4, 5, 6]))


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
def test_diff_exp_ttest(example_data, example_metadata, between_column, comparison, min_valid_values, expected_output):
    """Test diff_exp_ttest with various scenarios."""

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
            "max_level_1_samples": [5] * len(deltas),
            "max_level_2_samples": [5] * len(deltas),
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
@pytest.mark.skipif(not _HAS_ALPHAQUANT, reason="alphaquant not installed")
def test_diff_exp_alphaquant():
    """Testing function to ascertain stable functionality of diff_exp_alphaquant on small example datasets.

    The expected data were generated in alphapepttools/tests/tl/tl_test_data.ipynb and saved
    as .pkl files in alphapepttools/tests/tl/tl_test_data.

    """

    test_data_dir = Path(__file__).parent / "tl_test_data"
    report = pd.read_csv(test_data_dir / "example_dataset_mouse_sn_top20peptides.tsv", sep="\t")
    samplemap = pd.read_csv(test_data_dir / "samplemap_200.tsv", sep="\t")

    adata = ad.AnnData(
        X=pd.DataFrame(np.zeros(samplemap.shape[0]), index=samplemap["sample"], columns=["dummy"]),
        obs=samplemap.set_index("sample"),
    )

    # Mock raw AlphaQuant output (pre-standardization columns)
    mock_protein_df = pd.DataFrame(
        {
            "condition_pair": ["brain_VS_kidney", "brain_VS_kidney"],
            "protein": ["PROT1", "PROT2"],
            "log2fc": [1.0, -0.5],
            "p_value": [0.01, 0.05],
            "fdr": [0.02, 0.08],
            "quality_score": [0.9, 0.8],
        }
    )
    mock_proteoform_df = pd.DataFrame(
        {
            "protein": ["PROT1", "PROT2"],
            "log2fc": [1.0, -0.5],
            "proteoform_pval": [0.01, 0.05],
            "proteoform_fdr": [0.02, 0.08],
            "proteoform_id": ["PF1", "PF2"],
            "peptides": ["PEP1;PEP2", "PEP3"],
            "num_peptides": [2, 1],
            "quality_score": [0.9, 0.8],
        }
    )
    mock_peptide_df = pd.DataFrame(
        {
            "condition_pair": ["brain_VS_kidney", "brain_VS_kidney"],
            "protein": ["PROT1", "PROT2"],
            "log2fc": [1.0, -0.5],
            "p_value": [0.01, 0.05],
            "fdr": [0.02, 0.08],
            "sequence": ["SEQ_PEPTIDE1_", "SEQ_PEPTIDE2_"],
            "quality_score": [0.9, 0.8],
        }
    )

    # Build expected results (after standardization)
    expected_comparison_key = "brain_VS_kidney"
    expected_results = {
        "protein": pd.DataFrame(
            {
                "condition_pair": ["brain_VS_kidney", "brain_VS_kidney"],
                "protein": ["PROT1", "PROT2"],
                "log2fc": [1.0, -0.5],
                "p_value": [0.01, 0.05],
                "-log10(p_value)": [-np.log10(0.01), -np.log10(0.05)],  # Added by standardization
                "fdr": [0.02, 0.08],
                "-log10(fdr)": [-np.log10(0.02), -np.log10(0.08)],  # Added by standardization
                "method": ["alphaquant", "alphaquant"],  # Added by standardization
                "max_level_1_samples": [10, 10],  # Added by standardization
                "max_level_2_samples": [10, 10],  # Added by standardization
                "quality_score": [0.9, 0.8],
            }
        ),
        "proteoform": pd.DataFrame(
            {
                "condition_pair": ["brain_VS_kidney", "brain_VS_kidney"],
                "protein": ["PROT1", "PROT2"],
                "log2fc": [1.0, -0.5],
                "p_value": [0.01, 0.05],
                "-log10(p_value)": [-np.log10(0.01), -np.log10(0.05)],  # Added by standardization
                "fdr": [0.02, 0.08],
                "-log10(fdr)": [-np.log10(0.02), -np.log10(0.08)],  # Added by standardization
                "method": ["alphaquant", "alphaquant"],  # Added by standardization
                "max_level_1_samples": [10, 10],  # Added by standardization
                "max_level_2_samples": [10, 10],  # Added by standardization
                "proteoform_id": ["PF1", "PF2"],
                "peptides": ["PEP1;PEP2", "PEP3"],
                "num_peptides": [2, 1],
                "quality_score": [0.9, 0.8],
            }
        ),
        "peptide": pd.DataFrame(
            {
                "condition_pair": ["brain_VS_kidney", "brain_VS_kidney"],
                "protein": ["PROT1", "PROT2"],
                "log2fc": [1.0, -0.5],
                "p_value": [0.01, 0.05],
                "-log10(p_value)": [-np.log10(0.01), -np.log10(0.05)],  # Added by standardization
                "fdr": [0.02, 0.08],
                "-log10(fdr)": [-np.log10(0.02), -np.log10(0.08)],  # Added by standardization
                "method": ["alphaquant", "alphaquant"],  # Added by standardization
                "max_level_1_samples": [10, 10],  # Added by standardization
                "max_level_2_samples": [10, 10],  # Added by standardization
                "sequence": ["PEPTIDE1", "PEPTIDE2"],
                "quality_score": [0.9, 0.8],
            }
        ),
    }

    with (
        patch("alphapepttools.tl.diff_exp.alphaquant_wrapper.aq_pipeline.run_pipeline"),
        patch(
            "alphapepttools.tl.diff_exp.alphaquant_wrapper.pd.read_csv",
            side_effect=[mock_protein_df.copy(), mock_proteoform_df.copy(), mock_peptide_df.copy()],
        ),
    ):
        comparison_key, results = tl.diff_exp_alphaquant(
            adata=adata,
            report=report,
            between_column="condition",
            comparison=("brain", "kidney"),
            min_valid_values=2,
            valid_values_filter_mode="either",
            plots="hide",
        )

    assert comparison_key == expected_comparison_key

    for level in ["protein", "proteoform", "peptide"]:
        # The tolerances are slightly larger, than for the vanilla ttest, albeit still small, as the package is still in development
        pd.testing.assert_frame_equal(results[level], expected_results[level], rtol=0.01, atol=1e-6)


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


@pytest.fixture
def expected_ebayes_base_df():
    """Base expected dataframe for eBayes tests with B_VS_A comparison."""
    return pd.DataFrame(
        {
            "protein": ["X1", "X2", "X3"],
            "log2fc": [-0.7979892670672148, -2.8389205950315937, -1.5954559846999832],
            "p_value": [0.0030228412720276574, 6.564170711303982e-05, 0.002166002226930997],
            "-log10(p_value)": [2.5195846568222966, 4.182820132979508, 2.664341101199458],
            "fdr": [0.0030228412720276574, 0.00019692512133911947, 0.0030228412720276574],
            "-log10(fdr)": [2.5195846568222966, 3.7056988782598457, 2.5195846568222966],
            "method": ["limma_ebayes_inmoose", "limma_ebayes_inmoose", "limma_ebayes_inmoose"],
            "max_level_1_samples": [5, 5, 5],
            "max_level_2_samples": [5, 5, 5],
            "stat": [-3.804007666004946, -6.2764817445960475, -3.9999011197249166],
            "B": [-1.8736068846693623, 2.010219213410613, -1.5373419998901845],
            "AveExpr": [4.175828737355294, 2.8008384166375, 2.1791061114716954],
        },
        index=["X1", "X2", "X3"],
    )


# Test diff_exp_limma by loading small example datasets
@pytest.mark.skipif(not _HAS_INMOOSE, reason="inmoose not installed")
@pytest.mark.parametrize(
    ("comparison", "expected_comparison_key", "between_column"),
    [
        (("B", "A"), "B_VS_A", "group"),  # ensure that patsy's alphabetical ordering is cancelled out correctly
    ],
)
def test_diff_exp_ebayes(
    example_adata_ebayes,
    expected_ebayes_base_df,
    comparison,
    expected_comparison_key,
    between_column,
):
    """Testing function to ascertain stable functionality of diff_exp_limma on a small example dataset."""

    adata = example_adata_ebayes.copy()

    comparison_key, results = tl.diff_exp_ebayes(
        adata=adata,
        between_column=between_column,
        comparison=comparison,
    )

    # Add the condition_pair column to the expected dataframe
    expected_df = expected_ebayes_base_df.copy()
    expected_df.insert(0, "condition_pair", [expected_comparison_key] * len(expected_df))

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
                    "max_level_1_samples": [5],
                    "max_level_2_samples": [5],
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
                    "max_level_1_samples": [5],
                    "max_level_2_samples": [5],
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
                    "max_level_1_samples": [5],
                    "max_level_2_samples": [5],
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
                    "max_level_1_samples": [5],
                    "max_level_2_samples": [5],
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


### Expanded eBayes tests


# Critical test for the expanded implementation: on complete features it must reproduce
# the original diff_exp_ebayes exactly.
@pytest.mark.skipif(not _HAS_INMOOSE, reason="inmoose not installed")
@pytest.mark.parametrize(
    ("comparison", "expected_comparison_key", "between_column"),
    [
        (("B", "A"), "B_VS_A", "group"),
    ],
)
def test_diff_exp_ebayes_expanded_agrees_with_original(
    example_adata_ebayes,
    comparison,
    expected_comparison_key,
    between_column,
):
    """The nan-aware expanded eBayes must reproduce the original diff_exp_ebayes on shared features.

    The original drops any feature with a missing value, whereas the expanded version fits every feature.
    To match, we apply the same upfront completeness filter (the intended workflow) before the expanded
    version so both estimate the eBayes prior from the same feature set; the moderated statistics must then
    agree to numerical precision. We compare on the features the original returns and on the columns both
    implementations share (the expanded output lacks the original's extra `stat`, `B`, `AveExpr`, and carries
    a distinct `method` label).
    """
    adata = example_adata_ebayes.copy()

    # Original implementation: returns (comparison_key, DataFrame), drops incomplete features.
    comparison_key, original = tl.diff_exp_ebayes(
        adata=adata.copy(),
        between_column=between_column,
        comparison=comparison,
    )
    assert comparison_key == expected_comparison_key

    # Expanded implementation fits every feature, so filter incomplete features upfront (as a user would)
    # to match the original's feature set and therefore its eBayes prior.
    adata_complete = filter_data_completeness(adata.copy(), max_missing_count=0, action="drop")
    expanded_results = diff_exp_ebayes_expanded(
        adata=adata_complete,
        between_column=between_column,
        comparison=comparison,
    )
    assert set(expanded_results) == {expected_comparison_key}
    expanded = expanded_results[expected_comparison_key]

    # Restrict to the features the original returns and the columns both share (excluding `method`,
    # which is an intentionally distinct label rather than a computed result).
    compare_cols = [c for c in tl_defaults.DIFF_EXP_COLS if c != "method"]
    expanded_shared = expanded.loc[original.index, compare_cols]
    original_shared = original[compare_cols]

    pd.testing.assert_frame_equal(
        expanded_shared,
        original_shared,
        check_exact=False,
        rtol=1e-5,
        atol=1e-8,
        check_dtype=False,
        check_names=False,
    )

    # The method labels are intentionally distinct between the two implementations.
    assert original["method"].unique().tolist() == ["limma_ebayes_inmoose"]
    assert expanded["method"].unique().tolist() == ["limma_ebayes_inmoose_expanded"]


# Unit tests for the expanded eBayes components


def _abc_adata():
    """AnnData with three conditions A, B, C (two samples each); X is unused by the contrast helpers."""
    obs = pd.DataFrame(
        {"group": ["A", "A", "B", "B", "C", "C"]},
        index=[f"s{i}" for i in range(6)],
    )
    return ad.AnnData(X=np.zeros((6, 2), dtype=float), obs=obs)


# Test building a design matrix with and without covariates, and validate error handling for invalid inputs.
def test__build_design_matrix_basic():
    """Without a covariate, the design matrix is a one-hot encoding of the conditions."""
    obs = pd.DataFrame({"group": ["A", "A", "B", "B"]}, index=[f"s{i}" for i in range(4)])
    adata = ad.AnnData(X=np.zeros((4, 1), dtype=float), obs=obs)

    dm, col_info = _build_design_matrix(adata, "group")

    # Columns follow the order conditions first appear in.
    assert list(dm.columns) == ["A", "B"]
    assert list(dm.index) == list(adata.obs_names)
    np.testing.assert_array_equal(dm.to_numpy(), np.array([[1, 0], [1, 0], [0, 1], [0, 1]]))

    # Each row is one-hot across the condition columns.
    np.testing.assert_array_equal(dm.to_numpy().sum(axis=1), np.ones(4, dtype=int))
    assert col_info == {"condition_col_idxs": {"A": 0, "B": 1}, "covariate_col_idxs": {}}


def test__build_design_matrix_with_covariate():
    """A covariate is added in k-1 fashion: the first level is dropped to avoid multicollinearity."""
    obs = pd.DataFrame(
        {"group": ["A", "A", "B", "B"], "batch": ["x", "x", "y", "y"]},
        index=[f"s{i}" for i in range(4)],
    )
    adata = ad.AnnData(X=np.zeros((4, 1), dtype=float), obs=obs)

    dm, col_info = _build_design_matrix(adata, "group", covariate_column="batch")

    # Two condition columns plus one covariate column ("x", the first level, is dropped).
    assert list(dm.columns) == ["A", "B", "y"]
    np.testing.assert_array_equal(dm["y"].to_numpy(), np.array([0, 0, 1, 1]))
    assert col_info == {"condition_col_idxs": {"A": 0, "B": 1}, "covariate_col_idxs": {"y": 2}}


def test__build_design_matrix_column_order_follows_first_appearance():
    """Columns follow the order levels first appear in, not the lexicographic order get_dummies uses."""
    obs = pd.DataFrame(
        {"group": ["treated", "treated", "ctrl", "ctrl"], "batch": ["b", "a", "b", "a"]},
        index=[f"s{i}" for i in range(4)],
    )
    adata = ad.AnnData(X=np.zeros((4, 1), dtype=float), obs=obs)

    dm, col_info = _build_design_matrix(adata, "group", covariate_column="batch")

    # "treated" precedes "ctrl"; the covariate keeps "b" because "a" is the dropped first level.
    assert list(dm.columns) == ["treated", "ctrl", "b"]
    assert col_info["condition_col_idxs"] == {"treated": 0, "ctrl": 1}


def test__build_design_matrix_ignores_unused_categories():
    """Categories with no samples left (e.g. after subsetting) must not add all-zero columns."""
    obs = pd.DataFrame(
        {
            "group": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
            "batch": pd.Categorical(["x", "x", "y", "y"], categories=["x", "y", "z"]),
        },
        index=[f"s{i}" for i in range(4)],
    )
    adata = ad.AnnData(X=np.zeros((4, 1), dtype=float), obs=obs)

    dm, col_info = _build_design_matrix(adata, "group", covariate_column="batch")

    # Unobserved levels "C" and "z" would make the design matrix rank-deficient.
    assert list(dm.columns) == ["A", "B", "y"]
    assert col_info == {"condition_col_idxs": {"A": 0, "B": 1}, "covariate_col_idxs": {"y": 2}}


# Test raise behavior for invalid condition/covariate specifications in _build_design_matrix
@pytest.mark.parametrize(
    ("condition", "covariate", "between_column", "covariate_column"),
    [
        (["A", "A"], None, "missing", None),  # condition column absent
        ([np.nan, "A"], None, "group", None),  # NaN in condition column
        (["A", "B"], ["x", "x"], "group", "missing"),  # covariate column absent
        (["A", "B"], [np.nan, "x"], "group", "batch"),  # NaN in covariate column
    ],
)
def test__build_design_matrix_validation(condition, covariate, between_column, covariate_column):
    """Invalid condition/covariate specifications raise KeyError."""
    data = {"group": condition}
    if covariate is not None:
        data["batch"] = covariate
    obs = pd.DataFrame(data, index=[f"s{i}" for i in range(len(condition))])
    adata = ad.AnnData(X=np.zeros((len(condition), 1), dtype=float), obs=obs)

    with pytest.raises(KeyError):
        _build_design_matrix(adata, between_column, covariate_column=covariate_column)


# Nan-aware linear fit (counterpart to inmoose.limma.lmFit)
@pytest.fixture
def lmfit_adata():
    """Three features exercising the complete / control-missing / empty-condition fit paths.

    Samples s0-s2 are control "A", s3-s5 are treatment "B".
    """
    x = pd.DataFrame(
        {
            "complete": [2, 4, 6, 1, 2, 3],  # full data in both groups
            "control_missing": [2, 4, np.nan, 1, 2, 3],  # one missing control value
            "treat_all_missing": [2, 4, 6, np.nan, np.nan, np.nan],  # treatment group entirely missing
        },
        index=[f"s{i}" for i in range(6)],
    ).astype(float)
    obs = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"]}, index=[f"s{i}" for i in range(6)])
    return ad.AnnData(X=x, obs=obs)


# First check the case with complete features
def test__nan_lmfit_complete_feature(lmfit_adata):
    """For a fully observed feature the fit recovers group means, residual variance and df exactly."""
    design_matrix, col_info = _build_design_matrix(lmfit_adata, "group")
    fit = _nan_lmfit(lmfit_adata, design_matrix)
    j = list(lmfit_adata.var_names).index("complete")

    # Coefficients are the group means (A=4, B=2); condition order is [A, B].
    assert col_info["condition_col_idxs"] == {"A": 0, "B": 1}
    np.testing.assert_allclose(fit["B"][:, j], [4.0, 2.0])
    # SSR = 8 (A) + 2 (B) = 10, df = 6 - 2 = 4, sigma2 = 10/4.
    np.testing.assert_allclose(fit["dfs"][j], 4.0)
    np.testing.assert_allclose(fit["sigma2"][j], 2.5)
    # Unscaled covariance is pinv(X'X) = diag(1/n_A, 1/n_B) = diag(1/3, 1/3).
    np.testing.assert_allclose(fit["M_all"][j], np.array([[1 / 3, 0.0], [0.0, 1 / 3]]))


# Next check the case with empty condition columns (all missing in one group)
def test__nan_lmfit_drops_empty_condition_column(lmfit_adata):
    """A condition with no observed values is dropped and scattered back as NaN, the rest is fit."""
    design_matrix, _ = _build_design_matrix(lmfit_adata, "group")
    fit = _nan_lmfit(lmfit_adata, design_matrix)
    j = list(lmfit_adata.var_names).index("treat_all_missing")

    # Only the control mean is estimable; the dead treatment coefficient is NaN.
    np.testing.assert_allclose(fit["B"][:, j], [4.0, np.nan], equal_nan=True)
    # df = 3 - 1 = 2, SSR = 8, sigma2 = 4.
    np.testing.assert_allclose(fit["dfs"][j], 2.0)
    np.testing.assert_allclose(fit["sigma2"][j], 4.0)
    # Only the (A, A) entry of the unscaled covariance is populated.
    np.testing.assert_allclose(fit["M_all"][j], np.array([[1 / 3, np.nan], [np.nan, np.nan]]), equal_nan=True)


# Test making of contrasts from a design matrix, which are needed to compute the actual log2FC as [B_treatment - B_control] for each contrast.
@pytest.mark.parametrize(
    ("control_is", "expected"),
    [
        (1, np.array([[1, -1, 0], [1, 0, -1]])),  # control = +1, each treatment = -1
        (-1, np.array([[-1, 1, 0], [-1, 0, 1]])),  # control = -1, each treatment = +1
    ],
)
def test__make_contrasts(control_is, expected):
    """The contrast matrix has the control on every row and -control_is in each treatment's own row."""
    adata = _abc_adata()
    cm = _make_contrasts(adata, between_column="group", control_condition="A", control_is=control_is)

    # Columns are the conditions; rows are the K-1 treatment-vs-control contrasts.
    assert list(cm.columns) == ["A", "B", "C"]
    assert cm.shape == (2, 3)
    np.testing.assert_array_equal(cm.to_numpy(), expected)


# Test computing the contrast log2FC, unscaled variance and standard deviation from the contrast matrix into separate arrays
def test__run_contrasts():
    """log2fc and unscaled variance are computed per contrast, dropping covariate rows/cols."""
    # Conditions A, B, C at indices 0-2, plus a covariate at index 3 that must be ignored.
    col_info = {"condition_col_idxs": {"A": 0, "B": 1, "C": 2}, "covariate_col_idxs": {"cov": 3}}
    # One feature; condition coefficients [1, 3, 4] and a covariate coefficient (99) to be dropped.
    b = np.array([[1.0], [3.0], [4.0], [99.0]])
    # Unscaled covariance: identity on the conditions, large values on the covariate row/col.
    m = np.full((1, 4, 4), 1000.0)
    m[0, :3, :3] = np.eye(3)
    contrast_matrix = _make_contrasts(_abc_adata(), between_column="group", control_condition="A", control_is=1)

    out = _run_contrasts(contrast_matrix, B=b, M_all=m, col_info=col_info)

    # Contrast 0 = A - B = 1 - 3 = -2; contrast 1 = A - C = 1 - 4 = -3 (covariate coef ignored).
    np.testing.assert_allclose(out["log2fc"], np.array([[-2.0], [-3.0]]))
    # Quadratic form C @ I @ C = 2 for each contrast (covariate entries excluded by subsetting).
    np.testing.assert_allclose(out["unscaled_var"], np.array([[2.0], [2.0]]))
    np.testing.assert_allclose(out["stdev_unscaled"], np.sqrt(np.array([[2.0], [2.0]])))


def test__run_contrasts_matches_explicit_quadratic_form():
    """The einsum-based unscaled variance matches an explicit per-feature, per-contrast loop."""
    col_info = {"condition_col_idxs": {"A": 0, "B": 1, "C": 2}, "covariate_col_idxs": {}}
    contrast_matrix = _make_contrasts(_abc_adata(), between_column="group", control_condition="A", control_is=-1)
    c = contrast_matrix.to_numpy()

    # A few features with distinct, non-trivial (but symmetric) covariance matrices.
    rng_free = np.array([[2.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 3.0]])
    m = np.stack([rng_free, rng_free * 2.0, np.eye(3)])
    b = np.array([[1.0, 0.0, 2.0], [3.0, 1.0, 2.0], [4.0, 2.0, 2.0]])

    out = _run_contrasts(contrast_matrix, B=b, M_all=m, col_info=col_info)

    expected = np.empty((c.shape[0], m.shape[0]))
    for j in range(m.shape[0]):
        for contrast in range(c.shape[0]):
            expected[contrast, j] = c[contrast] @ m[j] @ c[contrast]
    np.testing.assert_allclose(out["unscaled_var"], expected)


# Test the extraction of contrast names from a contrast matrix, to return things like "A_VS_B" or "B_VS_A" depending on the sign of the log2fc and the row order of the matrix.
@pytest.mark.parametrize(
    ("control_is", "expected_names"),
    [
        (1, ["A_VS_B", "A_VS_C"]),  # log2fc = control - treatment -> "control_VS_treatment"
        (-1, ["B_VS_A", "C_VS_A"]),  # log2fc = treatment - control -> "treatment_VS_control"
    ],
)
def test__contrasts_from_matrix_naming(control_is, expected_names):
    """Contrast names follow the sign of log2fc and the row order of the matrix."""
    adata = _abc_adata()
    cm = _make_contrasts(adata, between_column="group", control_condition="A", control_is=control_is)

    assert _contrasts_from_matrix(cm, control_condition="A") == expected_names


@pytest.mark.parametrize(
    ("matrix", "control_condition", "expected_exception"),
    [
        # Control condition not present in the matrix columns.
        (pd.DataFrame([[1, -1, 0]], columns=["A", "B", "C"]), "Z", KeyError),
        # Control column appears more than once.
        (pd.DataFrame(np.array([[1, -1, 1]]), columns=["A", "B", "A"]), "A", ValueError),
        # A row with two non-zero treatment columns (not exactly one).
        (pd.DataFrame([[1, -1, -1]], columns=["A", "B", "C"]), "A", ValueError),
        # A row with an invalid sign pattern (both +1).
        (pd.DataFrame([[1, 1, 0]], columns=["A", "B", "C"]), "A", ValueError),
    ],
)
def test__contrasts_from_matrix_errors(matrix, control_condition, expected_exception):
    """Malformed contrast matrices are rejected: an absent control column raises KeyError, structural defects ValueError."""
    with pytest.raises(expected_exception):
        _contrasts_from_matrix(matrix, control_condition=control_condition)


# Condition-ordering robustness:
# The fit and scatter-back must key coefficients by condition name, never by assuming the control
# comes first or that conditions are contiguous. These tests use a non-control-first, interspersed
# order (B, A, C with A as the control) to guard against positional mix-ups.


@pytest.fixture
def interspersed_adata():
    """Conditions in interspersed, non-control-first order (B, A, C; A is the control).

    Each condition has a distinct per-feature mean (A~11, B~21, C~31) so coefficients can be
    checked by name. The second feature has condition C entirely missing.
    """
    x = pd.DataFrame(
        {
            "all_present": [20.0, 10.0, 30.0, 22.0, 12.0, 32.0],
            "C_missing": [20.0, 10.0, np.nan, 22.0, 12.0, np.nan],
        },
        index=[f"s{i}" for i in range(6)],
    )
    obs = pd.DataFrame({"group": ["B", "A", "C", "B", "A", "C"]}, index=[f"s{i}" for i in range(6)])
    return ad.AnnData(X=x, obs=obs)


def test__nan_lmfit_maps_coefficients_by_name_under_interspersed_order(interspersed_adata):
    """Coefficients align with conditions by name, not position, for arbitrary input ordering."""
    design_matrix, col_info = _build_design_matrix(interspersed_adata, "group")
    fit = _nan_lmfit(interspersed_adata, design_matrix)
    idx = col_info["condition_col_idxs"]

    # Appearance order is [B, A, C]: the control "A" is deliberately NOT the first column.
    assert idx == {"B": 0, "A": 1, "C": 2}

    j_all = list(interspersed_adata.var_names).index("all_present")
    # Each coefficient lands on its own condition (A=11, B=21, C=31).
    np.testing.assert_allclose(fit["B"][idx["A"], j_all], 11.0)
    np.testing.assert_allclose(fit["B"][idx["B"], j_all], 21.0)
    np.testing.assert_allclose(fit["B"][idx["C"], j_all], 31.0)

    j_miss = list(interspersed_adata.var_names).index("C_missing")
    # The dropped condition (C) scatters back to NaN at its own index; the others are unaffected.
    np.testing.assert_allclose(fit["B"][idx["A"], j_miss], 11.0)
    np.testing.assert_allclose(fit["B"][idx["B"], j_miss], 21.0)
    assert np.isnan(fit["B"][idx["C"], j_miss])
    # The NaN is confined to C's row/column of the unscaled covariance; the B/A block stays finite.
    m_miss = fit["M_all"][j_miss]
    assert np.isnan(m_miss[idx["C"], :]).all()
    assert np.isnan(m_miss[:, idx["C"]]).all()
    assert np.isfinite(m_miss[np.ix_([idx["B"], idx["A"]], [idx["B"], idx["A"]])]).all()


def test__run_contrasts_log2fc_correct_under_interspersed_order(interspersed_adata):
    """End-to-end through _run_contrasts: each contrast's log2fc is treatment - control, by name."""
    design_matrix, col_info = _build_design_matrix(interspersed_adata, "group")
    fit = _nan_lmfit(interspersed_adata, design_matrix)
    cm = _make_contrasts(interspersed_adata, between_column="group", control_condition="A", control_is=-1)
    out = _run_contrasts(cm, B=fit["B"], M_all=fit["M_all"], col_info=col_info)
    names = _contrasts_from_matrix(cm, control_condition="A")

    j_all = list(interspersed_adata.var_names).index("all_present")
    log2fc_by_name = {name: out["log2fc"][i, j_all] for i, name in enumerate(names)}

    # control_is=-1 -> treatment - control: B - A = 10, C - A = 20.
    np.testing.assert_allclose(log2fc_by_name["B_VS_A"], 10.0)
    np.testing.assert_allclose(log2fc_by_name["C_VS_A"], 20.0)


# Bit of finageling to skip the need for inmoose in this test, which we would need if we ran the entire
# pipeline of ebayes_expanded.diff_exp_ebayes. Instead, we mock the fit and contrast step and check the correct ordering
# of the results by name, which is what we are guarding against.
def test_fit_and_contrasts_invariant_to_sample_permutation(example_adata_ebayes):
    """Permuting the input samples must not change the per-contrast log2fc/variance (matched by name).

    The permutation flips the condition appearance order (B-first to A-first), which reorders the
    internal design and contrast rows; the named results must nonetheless be identical. This stops
    before the eBayes step so it runs without inmoose -- the scatter-back is what we are guarding.
    """

    def fit_and_contrasts_by_name(adata):
        design_matrix, col_info = _build_design_matrix(adata, "group")
        fit = _nan_lmfit(adata, design_matrix)
        cm = _make_contrasts(adata, between_column="group", control_condition="A", control_is=-1)
        out = _run_contrasts(cm, B=fit["B"], M_all=fit["M_all"], col_info=col_info)
        names = _contrasts_from_matrix(cm, control_condition="A")
        return {name: (out["log2fc"][i], out["unscaled_var"][i]) for i, name in enumerate(names)}

    adata = example_adata_ebayes.copy()
    base = fit_and_contrasts_by_name(adata)

    # A fixed permutation that interleaves the two groups (group becomes A, B, A, B, ...).
    perm = [5, 0, 7, 2, 9, 1, 6, 3, 8, 4]
    shuffled = fit_and_contrasts_by_name(adata[perm].copy())

    assert set(base) == set(shuffled)
    for name, (log2fc, unscaled_var) in base.items():
        np.testing.assert_allclose(shuffled[name][0], log2fc, equal_nan=True)
        np.testing.assert_allclose(shuffled[name][1], unscaled_var, equal_nan=True)


# The replicate gate suppresses a contrast's fold change when either side has too few observed values. The
# feature is still fit (and contributes to the eBayes prior); only the reported fold change/p/fdr are NaNed.
@pytest.fixture
def gate_adata():
    """Comparison ("X", "Y"): one feature sparse in X (2 of 5), one sparse in Y (2 of 5), the rest full."""
    x = pd.DataFrame(
        {
            "full_1": [10, 12, 14, 16, 18, 1, 2, 3, 4, 5],
            "full_2": [1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
            "full_3": [2, 4, 6, 8, 10, 1, 3, 5, 7, 9],
            "x_sparse": [np.nan, np.nan, np.nan, 16, 18, 1, 2, 3, 4, 5],  # X has only 2 observed
            "y_sparse": [10, 12, 14, 16, 18, 1, 2, np.nan, np.nan, np.nan],  # Y has only 2 observed
        },
        index=[f"cell{i}" for i in range(10)],
    ).astype(float)
    obs = pd.DataFrame({"group": ["X"] * 5 + ["Y"] * 5}, index=[f"cell{i}" for i in range(10)])
    adata = ad.AnnData(X=x, obs=obs)
    nanlog(adata)
    return adata


@pytest.mark.skipif(not _HAS_INMOOSE, reason="inmoose not installed")
@pytest.mark.parametrize(
    ("a_min_required", "sparse_reported"),
    [
        (3, False),  # only 2 observed A (X) values, below the required 3 -> fold change suppressed
        (2, True),  # 2 observed A (X) values meet the requirement -> fold change reported
        (None, True),  # gate disabled -> fold change reported
    ],
)
def test_diff_exp_ebayes_a_gate(gate_adata, a_min_required, sparse_reported):
    """a_min_required suppresses (NaNs) fold changes whose A condition has too few observed values."""
    results = diff_exp_ebayes_expanded(
        adata=gate_adata,
        between_column="group",
        comparison=("X", "Y"),
        a_min_required=a_min_required,
    )
    df = results["X_VS_Y"].set_index("protein")
    result_cols = ["log2fc", "p_value", "fdr"]

    # Fully observed features are always reported, regardless of the gate.
    assert df.loc["full_1", result_cols].notna().all()

    # The feature sparse in A (X) is reported only when the requirement admits it.
    sparse = df.loc["x_sparse", result_cols]
    if sparse_reported:
        assert sparse.notna().all()
    else:
        assert sparse.isna().all()


@pytest.mark.skipif(not _HAS_INMOOSE, reason="inmoose not installed")
@pytest.mark.parametrize(
    ("b_min_required", "sparse_reported"),
    [
        (3, False),  # only 2 observed B (Y) values, below the required 3 -> fold change suppressed
        (2, True),  # 2 observed B (Y) values meet the requirement -> fold change reported
        (None, True),  # gate disabled -> fold change reported
    ],
)
def test_diff_exp_ebayes_b_gate(gate_adata, b_min_required, sparse_reported):
    """b_min_required suppresses (NaNs) fold changes whose B condition has too few observed values."""
    results = diff_exp_ebayes_expanded(
        adata=gate_adata,
        between_column="group",
        comparison=("X", "Y"),
        b_min_required=b_min_required,
    )
    df = results["X_VS_Y"].set_index("protein")
    result_cols = ["log2fc", "p_value", "fdr"]

    # Fully observed features are always reported, regardless of the gate.
    assert df.loc["full_1", result_cols].notna().all()

    # The feature sparse in B (Y) is reported only when the requirement admits it.
    sparse = df.loc["y_sparse", result_cols]
    if sparse_reported:
        assert sparse.notna().all()
    else:
        assert sparse.isna().all()


# A covariate is modelled as an additive effect, so it must absorb a batch offset that is unevenly
# distributed across the conditions. Without it, that offset is confounded with the group effect.
_COVARIATE_TRUE_EFFECT = 1.0
_COVARIATE_MIN_UNADJUSTED_BIAS = 0.9
_COVARIATE_MAX_ADJUSTED_ERROR = 0.5


@pytest.fixture
def confounded_batch_adata():
    """Groups A/B with an unbalanced batch: A is mostly b1, B is mostly b2.

    Every feature carries the same true group effect, plus a feature-specific offset on the b2
    samples. Because batch is unevenly distributed across the groups, the unadjusted group
    difference measures the group effect plus most of that offset. Values are already additive,
    so no nanlog is applied.
    """
    sample_names = [f"s{i}" for i in range(8)]
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    batches = np.array(["b1", "b1", "b1", "b2", "b1", "b2", "b2", "b2"])

    group_effect = np.where(groups == "B", _COVARIATE_TRUE_EFFECT, 0.0)
    is_b2 = batches == "b2"

    rng = np.random.default_rng(0)
    features = {}
    for feature_idx in range(10):
        baseline = 10.0 + feature_idx  # feature-specific overall level
        batch_offset = 2.0 + 0.5 * feature_idx  # feature-specific size of the b2 effect
        noise_sd = 0.05 + 0.03 * feature_idx  # varies so residual variances stay heterogeneous

        values = baseline + group_effect
        values = values + np.where(is_b2, batch_offset, 0.0)
        values = values + rng.normal(0, noise_sd, size=len(sample_names))

        features[f"f{feature_idx}"] = values

    x = pd.DataFrame(features, index=sample_names)
    obs = pd.DataFrame({"group": groups, "batch": batches}, index=sample_names)
    return ad.AnnData(X=x, obs=obs)


@pytest.mark.skipif(not _HAS_INMOOSE, reason="inmoose not installed")
def test_diff_exp_ebayes_covariate_corrects_confounded_batch(confounded_batch_adata):
    """covariate_column absorbs the batch offset, recovering the true group effect it otherwise inflates."""
    unadjusted = diff_exp_ebayes_expanded(
        adata=confounded_batch_adata,
        between_column="group",
        comparison=("B", "A"),
    )["B_VS_A"]
    adjusted = diff_exp_ebayes_expanded(
        adata=confounded_batch_adata,
        between_column="group",
        comparison=("B", "A"),
        covariate_column="batch",
    )["B_VS_A"]

    # Adding a covariate must not change the output contract.
    assert list(adjusted.columns) == tl_defaults.DIFF_EXP_COLS
    assert adjusted.index.equals(confounded_batch_adata.var_names)

    unadjusted_error = (unadjusted["log2fc"] - _COVARIATE_TRUE_EFFECT).abs()
    adjusted_error = (adjusted["log2fc"] - _COVARIATE_TRUE_EFFECT).abs()

    # Ignoring the batch inflates every fold change; modelling it recovers the true effect.
    assert (unadjusted_error > _COVARIATE_MIN_UNADJUSTED_BIAS).all()
    assert (adjusted_error < _COVARIATE_MAX_ADJUSTED_ERROR).all()
    assert (adjusted_error < unadjusted_error).all()


# Comparison resolution turns the user-facing comparison tuple into explicit A conditions and the single B
# reference, expanding the "_ALL_" sentinel and validating every level up front.
@pytest.mark.parametrize(
    ("comparison", "expected_a", "expected_b"),
    [
        (("B", "A"), ["B"], "A"),  # single A condition is wrapped into a list
        ((["B", "C"], "A"), ["B", "C"], "A"),  # explicit list of A conditions passes through
        ((["C"], "A"), ["C"], "A"),  # single-element list needs no expansion
        (("_ALL_", "A"), ["B", "C"], "A"),  # sentinel expands to every level except B, in appearance order
    ],
)
def test__resolve_comparison(comparison, expected_a, expected_b):
    """Valid comparisons resolve to an explicit list of A conditions plus the B reference."""
    a_conditions, b_condition = _resolve_comparison(_abc_adata(), "group", comparison)

    assert isinstance(a_conditions, list)
    assert a_conditions == expected_a
    assert b_condition == expected_b


def test__resolve_comparison_all_sentinel_excludes_b():
    """ "_ALL_" never compares the B reference against itself, whichever level B is."""
    a_conditions, b_condition = _resolve_comparison(_abc_adata(), "group", ("_ALL_", "B"))

    assert b_condition not in a_conditions
    assert set(a_conditions) == {"A", "C"}


@pytest.mark.parametrize(
    ("between_column", "comparison"),
    [
        ("missing", ("B", "A")),  # between column absent from adata.obs
        ("group", ("B", "missing")),  # B reference is not a level of the between column
        ("group", ("missing", "A")),  # single A condition is not a level
        ("group", (["B", "missing"], "A")),  # one A condition of a list is not a level
        ("group", ("_ALL_", "missing")),  # sentinel cannot expand against an unknown B
    ],
)
def test__resolve_comparison_validation(between_column, comparison):
    """An unknown between column or condition raises KeyError before any fitting happens."""
    with pytest.raises(KeyError):
        _resolve_comparison(_abc_adata(), between_column, comparison)


# The replicate gate mask is the per-contrast AND of the two per-condition sufficiency masks; a None
# requirement disables that side of the gate.
@pytest.fixture
def gate_mask_adata():
    """Conditions X and Y (three samples each) with one feature sparse in X and one sparse in Y."""
    x = np.array(
        [
            # full, x_sparse, y_sparse
            [1.0, np.nan, 1.0],
            [2.0, np.nan, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, np.nan],
            [5.0, 5.0, np.nan],
            [6.0, 6.0, 6.0],
        ]
    )
    obs = pd.DataFrame({"group": ["X"] * 3 + ["Y"] * 3}, index=[f"s{i}" for i in range(6)])
    var = pd.DataFrame(index=["full", "x_sparse", "y_sparse"])
    return ad.AnnData(X=x, obs=obs, var=var)


@pytest.mark.parametrize(
    ("a_min_required", "b_min_required", "expected"),
    [
        (None, None, [True, True, True]),  # both gates disabled -> everything kept
        (2, None, [True, False, True]),  # A gate only: x_sparse has 1 observed in X
        (None, 2, [True, True, False]),  # B gate only: y_sparse has 1 observed in Y
        (2, 2, [True, False, False]),  # both gates -> both sparse features dropped
        (1, 1, [True, True, True]),  # requirements met everywhere
        (4, None, [False, False, False]),  # more required than X has samples -> nothing kept
    ],
)
def test__replicate_gate_mask(gate_mask_adata, a_min_required, b_min_required, expected):
    """The mask keeps a feature only where both conditions meet their (enabled) requirement."""
    keep = _replicate_gate_mask(
        gate_mask_adata,
        between_column="group",
        a_level="X",
        b_level="Y",
        a_min_required=a_min_required,
        b_min_required=b_min_required,
    )

    assert keep.dtype == bool
    assert keep.shape == (gate_mask_adata.n_vars,)
    np.testing.assert_array_equal(keep, np.array(expected))


# Output standardization is the single place the shared diff_exp column contract is applied to the
# expanded eBayes results, and the only place FDR correction happens.
_MAX_A_SAMPLES = 4
_MAX_B_SAMPLES = 5


def _contrast_frame_inputs():
    """A three-feature contrast: one significant, one not, one gated out (NaN)."""
    var_names = pd.Index(["p1", "p2", "p3"])
    log2fc = np.array([2.0, -0.5, np.nan])
    p_values = np.array([0.001, 0.5, np.nan])
    return var_names, log2fc, p_values


def test__standardize_contrast_frame_columns_follow_shared_contract():
    """The frame carries exactly the shared DIFF_EXP_COLS, in that order, indexed by feature."""
    var_names, log2fc, p_values = _contrast_frame_inputs()

    df = _standardize_contrast_frame(
        contrast_name="A_VS_B",
        var_names=var_names,
        log2fc=log2fc,
        p_values=p_values,
        max_level_1_samples=_MAX_A_SAMPLES,
        max_level_2_samples=_MAX_B_SAMPLES,
    )

    assert list(df.columns) == tl_defaults.DIFF_EXP_COLS
    assert df.index.equals(var_names)
    assert list(df["protein"]) == list(var_names)
    assert (df["condition_pair"] == "A_VS_B").all()
    assert (df["method"] == _METHOD_NAME).all()
    assert (df["max_level_1_samples"] == _MAX_A_SAMPLES).all()
    assert (df["max_level_2_samples"] == _MAX_B_SAMPLES).all()


def test__standardize_contrast_frame_derived_columns():
    """log2fc/p_value pass through untouched; fdr and both -log10 columns are derived from them."""
    var_names, log2fc, p_values = _contrast_frame_inputs()

    df = _standardize_contrast_frame(
        contrast_name="A_VS_B",
        var_names=var_names,
        log2fc=log2fc,
        p_values=p_values,
        max_level_1_samples=_MAX_A_SAMPLES,
        max_level_2_samples=_MAX_B_SAMPLES,
    )

    np.testing.assert_allclose(df["log2fc"].to_numpy(), log2fc, equal_nan=True)
    np.testing.assert_allclose(df["p_value"].to_numpy(), p_values, equal_nan=True)

    # FDR is the nan-safe BH correction of the (already gated) p-values
    np.testing.assert_allclose(df["fdr"].to_numpy(), tl.nan_safe_bh_correction(p_values), equal_nan=True)

    # Both -log10 columns mirror their source column
    np.testing.assert_allclose(df["-log10(p_value)"].to_numpy(), -np.log10(p_values), equal_nan=True)
    np.testing.assert_allclose(df["-log10(fdr)"].to_numpy(), -np.log10(df["fdr"].to_numpy()), equal_nan=True)


def test__standardize_contrast_frame_keeps_gated_features_nan():
    """A feature gated out upstream (NaN p-value) stays NaN across every derived column."""
    var_names, log2fc, p_values = _contrast_frame_inputs()

    df = _standardize_contrast_frame(
        contrast_name="A_VS_B",
        var_names=var_names,
        log2fc=log2fc,
        p_values=p_values,
        max_level_1_samples=_MAX_A_SAMPLES,
        max_level_2_samples=_MAX_B_SAMPLES,
    )

    derived_cols = ["log2fc", "p_value", "-log10(p_value)", "fdr", "-log10(fdr)"]
    assert df.loc["p3", derived_cols].isna().all()

    # The NaN feature must not consume a rank in the BH correction of the others
    np.testing.assert_allclose(df.loc[["p1", "p2"], "fdr"].to_numpy(), np.array([0.002, 0.5]))


# inmoose is an optional dependency; without it the failure must be an explicit ImportError rather than a
# NameError from inside the moderation step.
def test_diff_exp_ebayes_requires_inmoose():
    """diff_exp_ebayes raises ImportError up front when inmoose is unavailable."""
    with (
        patch("alphapepttools.tl.diff_exp.ebayes_expanded._HAS_INMOOSE", new=False),
        pytest.raises(ImportError, match="inmoose is required"),
    ):
        diff_exp_ebayes_expanded(adata=_abc_adata(), between_column="group", comparison=("B", "A"))
