import logging
import tempfile
from pathlib import Path

import alphaquant.run_pipeline as aq_pipeline
import anndata as ad
import pandas as pd

from alphapepttools.tl import tl_defaults
from alphapepttools.tl.utils import _suppress_plots, determine_max_replicates, negative_log10_pvalue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _standardize_alphaquant_results(
    comparison_key: str,
    level: str,
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize AlphaQuant result columns"""
    result_df = result_df.copy()

    # Base columns for all levels
    aq_columns = ["quality_score"]

    # Level-specific column mappings and extras
    if level == "protein":
        pval_column = "p_value"
        fdr_column = "fdr"
        extra_columns = []
        diff_exp_columns = [*tl_defaults.DIFF_EXP_COLS, *aq_columns]

    elif level == "proteoform":
        result_df["condition_pair"] = comparison_key
        pval_column = "proteoform_pval"
        fdr_column = "proteoform_fdr"
        extra_columns = ["proteoform_id", "peptides", "num_peptides"]
        diff_exp_columns = [*tl_defaults.DIFF_EXP_COLS, *extra_columns, *aq_columns]

    elif level == "peptide":
        pval_column = "p_value"
        fdr_column = "fdr"
        extra_columns = ["sequence"]
        diff_exp_columns = [*tl_defaults.DIFF_EXP_COLS, *extra_columns, *aq_columns]

        # Clean sequence names for peptides
        result_df["sequence"] = (
            result_df["sequence"].str.replace(r"^SEQ_", "", regex=True).str.replace(r"_$", "", regex=True)
        )

    else:
        raise ValueError(f"Unknown level: {level}. Must be 'protein', 'peptide', or 'proteoform'")

    # Common standardization logic
    result_df["method"] = "alphaquant"

    # For p-values of exactly 0, use a very large value instead of NaN
    result_df["-log10(p_value)"] = result_df[pval_column].apply(negative_log10_pvalue)
    result_df["-log10(fdr)"] = result_df[fdr_column].apply(negative_log10_pvalue)

    # Renaming to common names
    result_df = result_df.rename(
        columns={
            pval_column: "p_value",
            fdr_column: "fdr",
        }
    )

    # Reorder columns for uniform output
    return result_df[diff_exp_columns].copy()


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

    samplemap = _get_samplemap(adata, between_column, comparison)

    # Record maximum number of replicates per group
    max_samples_g1, max_samples_g2 = determine_max_replicates(adata, between_column, comparison[0], comparison[1])

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
            df["max_level_1_samples"] = max_samples_g1
            df["max_level_2_samples"] = max_samples_g2
            results[level] = _standardize_alphaquant_results(comparison_key, level, df)

        return comparison_key, results
