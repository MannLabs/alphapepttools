import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

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


def _standardize_limma_results(
    comparison_key: str,
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize Limma eBayes result columns

    To harmonize the output of Limma with other methods, we rename some columns and
    add columns for context

    Parameters
    ----------
    comparison_key : str
        Identifier for the comparison, e.g. "group1_VS_group2".
    result_df : pd.DataFrame
        DataFrame with ttest results.

    Returns
    -------
    pd.DataFrame
        DataFrame with Limma eBayes differential expression results.

    """
    current_result_df = result_df.copy()

    current_result_df["condition_pair"] = comparison_key
    current_result_df["protein"] = current_result_df.index
    current_result_df = current_result_df.rename(columns={"logFC": "log2fc", "P.Value": "p_value", "adj.P.Val": "fdr"})

    # Calculate -log10 transformations
    current_result_df["-log10(p_value)"] = -current_result_df["p_value"].apply(
        lambda x: np.nan if x == 0 else np.log10(x)
    )
    current_result_df["-log10(fdr)"] = -current_result_df["fdr"].apply(lambda x: np.nan if x == 0 else np.log10(x))
    current_result_df["method"] = "limma_ebayes"

    # Extra columns specific to Limma
    return_cols = [*DIFF_EXP_COLS, "t", "B", "AveExpr"]

    return current_result_df[return_cols].copy()


def diff_exp_limma(
    adata: ad.AnnData,
    between_column: str,
    comparison: tuple[str, str],
) -> tuple[str, dict[str, pd.DataFrame]]:
    """
    Run limma eBayes differential expression analysis via R script.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with expression data in .X and sample metadata in .obs
    between_column : str
        Column name in adata.obs containing the contrast levels
    comparison : tuple
        Two levels to compare, e.g. ('healthy', 'disease')

    Returns
    -------
    tuple
        comparison_key : str
            Formatted comparison string
        results : dict
            Dictionary with 'protein' key containing limma results DataFrame
    """
    # Validate inputs
    if between_column not in adata.obs.columns:
        raise ValueError(f"Column '{between_column}' not found in adata.obs")

    level1, level2 = comparison
    available_levels = adata.obs[between_column].unique()
    if level1 not in available_levels or level2 not in available_levels:
        raise ValueError(f"Comparison levels {comparison} not found in {between_column}. Available: {available_levels}")

    # Filter adata to only include samples from the comparison
    mask = adata.obs[between_column].isin([level1, level2])
    adata_subset = adata[mask, :].copy()

    # Remove proteins/peptides with any missing values (as per paper)
    adata_subset = adata_subset[:, ~pd.isna(adata_subset.X).any(axis=0)].copy()

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)

        # Prepare file paths
        expression_file = temp_dir / "expression.csv"
        design_file = temp_dir / "design.csv"
        output_file = temp_dir / "limma_results.csv"

        # Prepare expression matrix (proteins x samples)
        expr_df = adata_subset.to_df().T.reset_index()
        expr_df.to_csv(expression_file, index=False)

        # Prepare design matrix
        design_df = pd.DataFrame(
            {"sample": adata_subset.obs_names, "condition": adata_subset.obs[between_column].to_list()}
        ).set_index("sample")
        design_df.to_csv(design_file)

        # Format contrast string for limma
        contrast_string = f"{level1}_VS_{level2}"  # Note: level2 - level1 for consistent direction

        # Run R script
        cmd = [
            "Rscript",
            "/Users/vincenthbrennsteiner/Documents/mann_labs/_git_repositories/alphasite/alphasite/call_limma.R",
            str(expression_file),
            str(design_file),
            contrast_string,
            str(output_file),
        ]

        print(f"Running limma with contrast: {contrast_string}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603

        # Check for errors
        if result.returncode != 0:
            print("R script error output:")
            print(result.stderr)
            raise RuntimeError(f"Limma analysis failed: {result.stderr}")

        # Read results
        if not output_file.exists():
            raise FileNotFoundError(f"Expected output file not found: {output_file}")

        limma_results = pd.read_csv(output_file, index_col=0)

    # Format comparison key consistent with AlphaQuant style
    comparison_key = f"{level1}_VS_{level2}"

    # Package results in expected format
    results = {"protein": _standardize_limma_results(comparison_key, limma_results)}

    return comparison_key, results
