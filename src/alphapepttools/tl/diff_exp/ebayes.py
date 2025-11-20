# Empirical Bayes moderated ttest for differential expression using InMoose's Limma implementation
import logging

import anndata as ad
import numpy as np
import pandas as pd
import patsy
from inmoose import limma

from alphapepttools.tl import tl_defaults
from alphapepttools.tl.utils import (
    determine_max_replicates,
    drop_features_with_too_few_valid_values,
    negative_log10_pvalue,
    validate_ttest_inputs,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _generate_patsy_design_matrix(
    adata_subset: ad.AnnData,
    between_column: str,
    level_1: str,
    level_2: str,
) -> tuple:
    """Generate Patsy design matrix with correct column ordering

    Patsy automatically sorts columns alphabetically, which can cause issues with
    contrast specifications. This function creates a design matrix and returns
    the column names in the desired order (level_1, level_2).

    Parameters
    ----------
    adata_subset : ad.AnnData
        AnnData object filtered to only include samples from the comparison
    between_column : str
        Column name in adata.obs containing the contrast levels
    level_1 : str
        First level (treatment)
    level_2 : str
        Second level (control)

    Returns
    -------
    tuple
        Design matrix and ordered column names

    """
    # Create design dataframe with condition as categorical factor
    design_df = pd.DataFrame(
        {"sample": adata_subset.obs_names, "condition": adata_subset.obs[between_column].to_list()}
    ).set_index("sample")

    # Create design matrix without intercept (fit means model: ~ 0 + condition)
    design_matrix = patsy.dmatrix("~ 0 + condition", data=design_df)

    # Patsy insists on changing the column order to alphabetical, so we need to resort them
    design_colnames = design_matrix.design_info.column_names
    pure_patsy_order = [x.replace("condition[", "").replace("]", "") for x in design_colnames]
    condition_order = [level_1, level_2]
    sorted_indices = [pure_patsy_order.index(cond) for cond in condition_order]
    design_colnames = [design_colnames[i] for i in sorted_indices]

    return design_matrix, design_colnames


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
    result_df = result_df.copy()
    diff_exp_columns = tl_defaults.DIFF_EXP_COLS.copy()

    # Map columns from Limma to standard names
    current_column_map = {"log2FoldChange": "log2fc", "pvalue": "p_value", "adj_pvalue": "fdr"}
    result_df = result_df.rename(columns=current_column_map)

    # Add standard columns
    result_df["condition_pair"] = comparison_key
    result_df["protein"] = result_df.index
    result_df["method"] = "limma_ebayes_inmoose"

    # For p-values of exactly 0, use a very large value instead of NaN
    result_df["-log10(p_value)"] = result_df["p_value"].apply(negative_log10_pvalue)
    result_df["-log10(fdr)"] = result_df["fdr"].apply(negative_log10_pvalue)

    # Extra columns specific to Limma
    return_cols = [*diff_exp_columns, "stat", "B", "AveExpr"]

    result_df.index.name = None

    return result_df[return_cols]


def diff_exp_ebayes(
    adata: ad.AnnData,
    between_column: str,
    comparison: tuple[str, str],
    min_valid_values: int = 2,
) -> tuple[str, pd.DataFrame]:
    """Run Limma eBayes moderated ttest for differential expression

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with expression data in .X and sample metadata in .obs
    between_column : str
        Column name in adata.obs containing the contrast levels
    comparison : tuple
        Two levels to compare, meant to follow the convention (treatment, control)
    min_valid_values : int, optional
        Minimum number of valid (non-missing) values required in at least one group. By default 2.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with Limma eBayes differential expression results.

    """
    logger.info("Running Limma eBayes differential expression analysis via inmoose")

    # Validate inputs
    level_1, level_2 = validate_ttest_inputs(adata, between_column, comparison, min_valid_values)
    logger.info(f"Comparing levels: {level_1} (treatment) vs {level_2} (control)")

    adata = drop_features_with_too_few_valid_values(
        adata,
        between_column=between_column,
        comparison=comparison,
        min_valid_values=min_valid_values,
    )

    # Filter adata to only include samples from the comparison
    adata_subset = adata[adata.obs[between_column].isin([level_1, level_2]), :].copy()

    # Report on maximum samples per level
    max_samples_level_1, max_samples_level_2 = determine_max_replicates(adata, between_column, level_1, level_2)

    # Generate design matrix with correct column ordering
    design_matrix, design_colnames = _generate_patsy_design_matrix(adata_subset, between_column, level_1, level_2)

    # Format a contrast string, which is required by the Limma differential expression function below
    contrast_string = f"{design_colnames[0]}-{design_colnames[1]}"
    logger.info(f"Computing contrast: {contrast_string}")

    # Expression matrix: proteins (rows) x samples (columns)
    expr_matrix = adata_subset.X.T

    logger.info(f"Design matrix dimensions: {design_matrix.shape[0]} samples x {design_matrix.shape[1]} groups")
    logger.info(f"Expression matrix dimensions: {expr_matrix.shape[0]} proteins x {expr_matrix.shape[1]} samples")

    # Initial linear fit (lmFit equivalent)
    fit = limma.lmFit(expr_matrix, design_matrix)

    # Contrast matrix: treatment - control
    contrast_matrix = limma.makeContrasts(contrast_string, levels=design_colnames)

    # Fit contrasts (contrasts.fit equivalent)
    fit2 = limma.contrasts_fit(fit, contrast_matrix)

    # Bayesian adjustment (eBayes equivalent)
    fit2 = limma.eBayes(fit2)

    # Extract results (topTable equivalent)
    contrast_name = fit2.coefficients.columns[0]

    # Toptable seems to be unable to accept sort_by="none", but sorts by pvalue by default. We have
    # to revert the sort by ordering by numeric feature index after extraction and then reassign the
    # features. Otherwise, the index will not match the original adata.var_names.
    toptable = limma.topTable(fit2, coef=contrast_name, number=np.inf)
    result_df = pd.DataFrame(toptable).sort_index()
    result_df.index = adata_subset.var_names

    # Add information on maximum samples per level
    result_df["max_level_1_samples"] = max_samples_level_1
    result_df["max_level_2_samples"] = max_samples_level_2

    comparison_key = f"{level_1}_VS_{level_2}"
    return comparison_key, _standardize_limma_results(comparison_key, result_df)
