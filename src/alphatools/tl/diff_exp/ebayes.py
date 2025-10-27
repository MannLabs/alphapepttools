# Empirical Bayes moderated ttest for differential expression using InMoose's Limma implementation
import logging

import anndata as ad
import numpy as np
import pandas as pd
import patsy
from inmoose import limma

from alphatools.tl import tl_defaults
from alphatools.tl.utils import drop_features_with_too_few_valid_values, validate_ttest_inputs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _standardize_limma_results(
    comparison_key: str,
    result_df: pd.DataFrame,
    adata: ad.AnnData = None,
    var_columns: list[str] | None = None,
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
    diff_exp_columns = tl_defaults.DIFF_EXP_COLS.copy()

    # Map columns from Limma to standard names
    current_column_map = {"log2FoldChange": "log2fc", "pvalue": "p_value", "adj_pvalue": "fdr"}
    current_result_df = current_result_df.rename(columns=current_column_map)

    # Add standard columns
    current_result_df["condition_pair"] = comparison_key
    current_result_df["protein"] = current_result_df.index
    current_result_df["method"] = "limma_ebayes_inmoose"

    # Calculate -log10 transformations
    current_result_df["-log10(p_value)"] = -current_result_df["p_value"].apply(
        lambda x: np.nan if x == 0 else np.log10(x)
    )
    current_result_df["-log10(fdr)"] = -current_result_df["fdr"].apply(lambda x: np.nan if x == 0 else np.log10(x))

    # Add gene annotation from AnnData var
    if var_columns is not None:
        var_columns = [var_columns] if not isinstance(var_columns, list) else var_columns
        if not all(col in adata.var.columns for col in var_columns):
            raise ValueError(f"Not all var columns {var_columns} are present in adata.var")
        current_result_df = current_result_df.join(adata.var[var_columns], how="left")
        diff_exp_columns = diff_exp_columns + var_columns

    # Extra columns specific to Limma
    return_cols = [*diff_exp_columns, "stat", "B", "AveExpr", "max_level_1_samples", "max_level_2_samples"]

    current_result_df.index.name = None

    return current_result_df[return_cols].copy()


def diff_exp_limma(
    adata: ad.AnnData,
    between_column: str,
    comparison: tuple[str, str],
    min_valid_values: int = 2,
    var_columns: list[str] | None = None,
) -> tuple[str, dict[str, pd.DataFrame]]:
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

    # Drop features with too few valid values
    adata = drop_features_with_too_few_valid_values(
        adata,
        between_column=between_column,
        comparison=comparison,
        min_valid_values=min_valid_values,
    )

    # Filter adata to only include samples from the comparison
    adata_subset = adata[adata.obs[between_column].isin([level_1, level_2]), :].copy()

    # Report on maximum samples per level
    max_samples_level_1 = adata.obs[adata.obs[between_column] == level_1].shape[0]
    max_samples_level_2 = adata.obs[adata.obs[between_column] == level_2].shape[0]

    logger.info(f"Number of samples for {level_1}: {max_samples_level_1}")

    logger.info(f"Number of samples for {level_2}: {max_samples_level_2}")

    # Create design dataframe with condition as categorical factor
    design_df = pd.DataFrame(
        {"sample": adata_subset.obs_names, "condition": adata_subset.obs[between_column].to_list()}
    ).set_index("sample")

    # Create design matrix without intercept (fit means model: ~ 0 + condition)
    design_matrix = patsy.dmatrix("~ 0 + condition", data=design_df)

    # Patsy changes the column names, so the new ones need to be used going forward
    design_colnames = design_matrix.design_info.column_names

    # Expression matrix: proteins (rows) x samples (columns)
    expr_matrix = adata_subset.X.T  # Transpose so proteins are rows

    # Report on progress
    logger.info(f"Design matrix dimensions: {design_matrix.shape[0]} samples x {design_matrix.shape[1]} groups")
    logger.info(f"Expression matrix dimensions: {expr_matrix.shape[0]} proteins x {expr_matrix.shape[1]} samples")

    # Format a contrast string for Limma
    contrast_string = f"{design_colnames[1]}-{design_colnames[0]}"
    logger.info(f"Computing contrast: {contrast_string}")

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

    # Toptable seems to be unable to accept sort_by="none", and sort by pvalue by default. We have
    # to revert the sort by ordering by numeric feature index after extraction and then reassign the
    # features.
    toptable = limma.topTable(fit2, coef=contrast_name, number=np.inf)
    result_df = pd.DataFrame(toptable).sort_index()
    result_df.index = adata_subset.var_names

    # Add information on maximum samples per level
    result_df["max_level_1_samples"] = max_samples_level_1
    result_df["max_level_2_samples"] = max_samples_level_2

    comparison_key = f"{level_1}_VS_{level_2}"
    return comparison_key, _standardize_limma_results(comparison_key, result_df, adata=adata, var_columns=var_columns)
