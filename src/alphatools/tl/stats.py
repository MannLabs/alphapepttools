import contextlib
import logging
import tempfile
from pathlib import Path

import alphaquant.run_pipeline as aq_pipeline
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ttest_ind

from alphatools.pp.data import filter_by_metadata

# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIFF_EXP_COLS = [
    "condition_pair",
    "protein",
    "log2fc",
    "p_value",
    "-log10(p_value)",
    "fdr",
    "-log10(fdr)",
    "method",
]


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
    min_valid_values: int | None = 2,
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


def _standardize_diff_exp_ttest_results(
    comparison_key: str,
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize ttest results DataFrames"""
    current_result_df = result_df.copy()

    # Map columns from ttest output format to standard format
    current_column_map = {
        f"delta_{comparison_key}": "log2fc",
        f"pvalue_{comparison_key}": "p_value",
        f"padj_{comparison_key}": "fdr",
    }

    # Select and rename columns
    columns_to_keep = [col for col in current_column_map if col in current_result_df.columns]
    current_result_df = current_result_df[columns_to_keep].rename(columns=current_column_map)

    # Add standard columns
    current_result_df["condition_pair"] = comparison_key
    current_result_df["protein"] = current_result_df.index
    current_result_df["method"] = "ttest"

    # Calculate -log10 transformations
    current_result_df["-log10(p_value)"] = -current_result_df["p_value"].apply(
        lambda x: np.nan if x == 0 else np.log10(x)
    )
    current_result_df["-log10(fdr)"] = -current_result_df["fdr"].apply(lambda x: np.nan if x == 0 else np.log10(x))

    # Reorder columns to match DIFF_EXP_COLS
    return current_result_df[DIFF_EXP_COLS].copy()


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
    result_df = pd.DataFrame(
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
    ).set_index("id", drop=True)
    result_df.index.name = None

    return _standardize_diff_exp_ttest_results(comparison_name, result_df)


def _standardize_alphaquant_results(
    comparison_key: str,
    level: str,
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize AlphaQuant result columns"""
    current_result_df = result_df.copy()

    # Base columns for all levels
    aq_columns = ["quality_score"]

    # Level-specific column mappings and extras
    if level == "protein":
        pval_column = "p_value"
        fdr_column = "fdr"
        extra_columns = []
        diff_exp_columns = [*DIFF_EXP_COLS, *aq_columns]

    elif level == "proteoform":
        current_result_df["condition_pair"] = comparison_key
        pval_column = "proteoform_pval"
        fdr_column = "proteoform_fdr"
        extra_columns = ["proteoform_id", "peptides", "num_peptides"]
        diff_exp_columns = [*DIFF_EXP_COLS, *extra_columns, *aq_columns]

    elif level == "peptide":
        pval_column = "p_value"
        fdr_column = "fdr"
        extra_columns = ["sequence"]
        diff_exp_columns = [*DIFF_EXP_COLS, *extra_columns, *aq_columns]

        # Clean sequence names for peptides
        current_result_df["sequence"] = (
            current_result_df["sequence"].str.replace(r"^SEQ_", "", regex=True).str.replace(r"_$", "", regex=True)
        )

    else:
        raise ValueError(f"Unknown level: {level}. Must be 'protein', 'peptide', or 'proteoform'")

    # Common standardization logic
    current_result_df["method"] = "alphaquant"
    current_result_df["-log10(p_value)"] = -current_result_df[pval_column].apply(
        lambda x: np.nan if x == 0 else np.log10(x)
    )
    current_result_df["-log10(fdr)"] = -current_result_df[fdr_column].apply(lambda x: np.nan if x == 0 else np.log10(x))

    # Renaming to common names
    current_result_df = current_result_df.rename(
        columns={
            pval_column: "p_value",
            fdr_column: "fdr",
        }
    )

    # Reorder columns for uniform output
    return current_result_df[diff_exp_columns].copy()


def diff_exp_alphaquant(
    adata: ad.AnnData,
    report: pd.DataFrame,
    between_column: str,
    comparison: tuple,
    min_valid_values: int = 2,
    valid_values_filter_mode: str = "either",
    plots: str = "hide",
) -> tuple[str, dict[str, pd.DataFrame]]:
    """Calculate differential expression using AlphaQuant."""
    if plots not in {"hide", "show"}:
        raise ValueError("Parameter 'plots' must be either 'hide' or 'show'.")

    # check that between column exists in obs
    if between_column not in adata.obs.columns:
        raise ValueError(f"Error in group_ratios_ttest_ind: Column '{between_column}' not found in adata.obs.")

    # validate comparison tuple
    if not isinstance(comparison, tuple) or len(comparison) != 2:  # noqa: PLR2004 since this is a tuple check
        raise ValueError("Error in group_ratios_ttest_ind: 'comparison' must be a tuple of exactly two group names.")

    # Generate samplemap in AlphaQuant format
    def _get_samplemap(
        adata: ad.AnnData,
        between_column: str,
        comparison: list,
    ) -> pd.DataFrame:
        """Extract AlphaQuant-compatible samplemap from AnnData.

        Returns DataFrame with 'sample' and 'condition' columns for the specified comparison groups.
        """
        mask = adata.obs[between_column].isin(comparison)
        return pd.DataFrame(
            {"sample": adata.obs.index[mask].astype(str), "condition": adata.obs.loc[mask, between_column]}
        ).reset_index(drop=True)

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

    samplemap = _get_samplemap(adata, between_column, comparison)

    # Context manager for AlphaQuant interface
    # For now, run with tempfiles TODO: PR on AlphaQuant for simplified interface
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)

        report_dir = temp_dir / "report.parquet"
        samplemap_dir = temp_dir / "samplemap.tsv"
        output_dir = temp_dir / "output"

        report.to_parquet(report_dir)
        samplemap.to_csv(samplemap_dir, sep="\t", index=False)

        # catch the plots
        if plots == "hide":
            with _suppress_plots():
                aq_pipeline.run_pipeline(
                    input_file=str(report_dir),
                    samplemap_file=str(samplemap_dir),
                    condpairs_list=[comparison],
                    results_dir=str(output_dir),
                    min_valid_values=min_valid_values,
                    valid_values_filter_mode=valid_values_filter_mode,
                )
        else:
            aq_pipeline.run_pipeline(
                input_file=str(report_dir),
                samplemap_file=str(samplemap_dir),
                condpairs_list=[comparison],
                results_dir=str(output_dir),
                min_valid_values=min_valid_values,
                valid_values_filter_mode=valid_values_filter_mode,
            )

        # Read AlphaQuant results for the single comparison
        # TODO: Parse precursor data from json tree? (https://github.com/MannLabs/alphaquant/issues/108)
        comparison_key = f"{comparison[0]}_VS_{comparison[1]}"

        results = {
            "protein": pd.read_csv(output_dir / f"{comparison_key}.results.tsv", sep="\t"),
            "proteoform": pd.read_csv(output_dir / f"{comparison_key}.proteoforms.tsv", sep="\t"),
            "peptide": pd.read_csv(output_dir / f"{comparison_key}.results.seq.tsv", sep="\t"),
        }

        # Standardize the result appearance
        for level, df in results.items():
            results[level] = _standardize_alphaquant_results(comparison_key, level, df)

        return comparison_key, results
