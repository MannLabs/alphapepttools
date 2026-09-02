import logging
import tempfile
from pathlib import Path

import anndata as ad
import pandas as pd

try:
    import alphaquant.run_pipeline as aq_pipeline

    _HAS_ALPHAQUANT = True
except ModuleNotFoundError:  # pragma: no cover - optional dep, env-dependent
    _HAS_ALPHAQUANT = False

from alphapepttools.tl import tl_defaults
from alphapepttools.tl.utils import _suppress_plots, determine_max_replicates, negative_log10_pvalue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Column holding each row's own identifier, per analysis level
_FEATURE_COLUMN_PER_LEVEL = {"protein": "protein", "proteoform": "proteoform_id", "peptide": "sequence"}

# Column order of the stacked output; level-specific columns are NaN outside their own modality
_STACKED_COLS = [
    "modality",
    "feature_id",
    *tl_defaults.DIFF_EXP_COLS,
    "quality_score",
    "proteoform_id",  # proteoform only
    "peptides",  # proteoform only
    "num_peptides",  # proteoform only
    "sequence",  # peptide only
]


def _standardize_alphaquant_results(
    comparison_key: str,
    level: str,
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize AlphaQuant result columns.

    Processes AlphaQuant output to create a standardized format with consistent
    column names and additional calculated fields. Handles different output formats
    for protein, peptide, and proteoform levels by mapping level-specific column
    names to standard names and adding negative log-transformed p-values.

    Parameters
    ----------
    comparison_key
        String identifying the comparison (e.g., "Group1_VS_Group2").
    level
        Analysis level - must be 'protein', 'peptide', or 'proteoform'.
    result_df
        Raw AlphaQuant results DataFrame to standardize.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with consistent column names and calculated fields
        including -log10 transformed p-values and FDR values.

    Raises
    ------
    ValueError
        If level is not one of 'protein', 'peptide', or 'proteoform'.
    """
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


def _stack_alphaquant_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack the per-level AlphaQuant results into a single long DataFrame.

    Concatenates the standardized protein, proteoform and peptide results into one
    frame that is filtered by the `modality` column instead of a dictionary key. Each
    row additionally carries its own identifier in the `feature_id` column, taken from the
    column appropriate for its level. Columns that only exist at one level are NaN on
    the rows of the other levels.

    Parameters
    ----------
    results
        Dictionary of standardized results keyed by 'protein', 'proteoform' and 'peptide'.

    Returns
    -------
    pd.DataFrame
        Stacked DataFrame with `modality` and `feature_id` columns and a uniform column order.
    """
    stacked = []
    for level, df in results.items():
        level_df = df.copy()
        level_df["modality"] = level
        level_df["feature_id"] = level_df[_FEATURE_COLUMN_PER_LEVEL[level]]
        stacked.append(level_df)

    # reindex both enforces the column order and pads the level-specific columns with NaN
    return pd.concat(stacked, axis=0, ignore_index=True).reindex(columns=_STACKED_COLS)


def diff_exp_alphaquant(
    adata: ad.AnnData,
    report: pd.DataFrame,
    between_column: str,
    comparison: tuple,
    min_valid_values: int = 2,
    valid_values_filter_mode: str = "either",
    plots: str = "hide",
) -> pd.DataFrame:
    """Calculate differential expression using AlphaQuant.

    Parameters
    ----------
    adata
        AnnData object containing the expression data and sample metadata.
    report
        DataFrame with quantification report data for AlphaQuant analysis.
    between_column
        Column name in adata.obs containing group labels for comparison.
    comparison
        Tuple of exactly two group names to compare.
    min_valid_values
        Minimum number of valid values required per group for statistical testing.
    valid_values_filter_mode
        How to apply the min_valid_values filter - 'either' or 'both'.
    plots
        Whether to 'show' or 'hide' AlphaQuant's generated plots.

    Returns
    -------
    pd.DataFrame
        Standardized results for all three analysis levels stacked into a single frame.
        The `modality` column holds 'protein', 'proteoform' or 'peptide' and is used to
        select the level of interest; the `feature_id` column holds each row's own
        identifier (protein accession, proteoform id or peptide sequence) and is never
        NaN. The comparison key (e.g. "Group1_VS_Group2") is carried in the
        `condition_pair` column.

    Notes
    -----
    Columns that only exist at one level (`sequence` for peptides, `proteoform_id`,
    `peptides` and `num_peptides` for proteoforms) are NaN on the rows of the other
    levels. As a consequence of stacking, `num_peptides` is of dtype float64 rather
    than int64.

    Raises
    ------
    ImportError
        If alphaquant is not installed.
    ValueError
        If plots is not 'hide' or 'show', if between_column is not in adata.obs,
        or if comparison is not a tuple of exactly two elements.

    Examples
    --------
    Run differential expression analysis between treatment groups:

    .. code-block:: python

        alphaquant_results = at.tl.diff_exp_alphaquant(
            adata=adata_precursor,
            report=full_report,
            between_column="treatment",
            comparison=("control", "treated"),
            valid_values_filter_mode="either",
            min_valid_values=3,
            plots="hide",
        )

        # Access results for different levels
        protein_results = alphaquant_results[alphaquant_results["modality"] == "protein"]
        peptide_results = alphaquant_results[alphaquant_results["modality"] == "peptide"]
        proteoform_results = alphaquant_results[alphaquant_results["modality"] == "proteoform"]

    """
    if not _HAS_ALPHAQUANT:  # pragma: no cover - optional dep, env-dependent
        raise ImportError(
            "alphaquant is required for diff_exp_alphaquant(). Install it through pip or install alphapepttools with the 'full'/'full-stable' extra."
        )

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
        comparison: list | tuple,
    ) -> pd.DataFrame:
        """Extract AlphaQuant-compatible samplemap from AnnData.

        Returns DataFrame with 'sample' and 'condition' columns for the specified comparison groups.

        Parameters
        ----------
        adata
            AnnData object containing sample metadata.
        between_column
            Column name in adata.obs containing group labels.
        comparison
            List of group names to include in the samplemap.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'sample' and 'condition' columns.
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
        else:  # pragma: no cover - plots="show" branch, only differs by skipping _suppress_plots wrapper
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

        return _stack_alphaquant_results(results)
