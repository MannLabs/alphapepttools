# Expanded eBayes differential expression module with multiple contrasts, nan-handling and covariate support

import anndata as ad
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from inmoose import limma

    _HAS_INMOOSE = True
except ModuleNotFoundError:
    _HAS_INMOOSE = False

from alphapepttools.tl.stats import nan_safe_bh_correction
from alphapepttools.tl.utils import (
    determine_max_replicates,
    negative_log10_pvalue,
)


def build_design_matrix(
    adata: ad.AnnData,
    condition_column: str,
    covariate_column: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build a design matrix for linear modeling.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing the data.
    condition_column : str
        The name of the column in adata.obs that contains the condition labels.
    covariate_column : str | None, optional
        The name of the column in adata.obs that contains the covariate labels. If None, no covariate columns are added.

    Returns
    -------
    pd.DataFrame
        A design matrix suitable for linear modeling, where columns are features and rows are samples. Each condition is encoded as a separate column. Covariate levels are added as additional columns in a k-1 fashion (one-hot encoding with one level dropped to avoid multicollinearity).
    dict
        A dictionary to navigate the design matrix with the following keys:
        - condition_col_idxs: A dictionary mapping each condition to its corresponding column index in the design matrix.
        - covariate_col_idxs: A dictionary mapping each covariate level to its corresponding column index in the design matrix. Empty if covariate_column is None.

    """
    if condition_column not in adata.obs.columns:
        raise ValueError(f"Condition column '{condition_column}' not found in adata.obs.")

    if adata.obs[condition_column].isna().any():
        raise ValueError(f"Condition column '{condition_column}' contains NaN values.")

    if covariate_column is not None:
        if covariate_column not in adata.obs.columns:
            raise ValueError(f"Covariate column '{covariate_column}' not found in adata.obs.")
        if adata.obs[covariate_column].isna().any():
            raise ValueError(f"Covariate column '{covariate_column}' contains NaN values.")

    condition_names = list(adata.obs[condition_column].unique())
    covariate_names = (
        list(adata.obs[covariate_column].unique()[:-1]) if covariate_column is not None else []
    )  # k-1 for covariates

    nrows = adata.n_obs
    ncols = len(condition_names) + len(covariate_names)

    dm = pd.DataFrame(
        data=np.zeros((nrows, ncols), dtype=int), index=adata.obs_names, columns=condition_names + covariate_names
    )

    for condition in condition_names:
        dm.loc[adata.obs[condition_column] == condition, condition] = 1

    for covariate in covariate_names:
        dm.loc[adata.obs[covariate_column] == covariate, covariate] = 1

    condition_col_idxs = {name: dm.columns.get_loc(name) for name in condition_names}
    covariate_col_idxs = {name: dm.columns.get_loc(name) for name in covariate_names}

    return dm, {"condition_col_idxs": condition_col_idxs, "covariate_col_idxs": covariate_col_idxs}


def summarize_design_matrix(
    dm: pd.DataFrame,
) -> None:
    """Summarize the design matrix by reporting the counts of samples for each condition and covariate level."""
    counts = dm.sum(axis=0)
    columns = dm.columns.tolist()
    order = np.argsort(counts.values)[::-1]
    print("Design Matrix Summary:")
    for col in np.array(columns)[order]:
        print(f"{col}: {counts[col]}")


def nan_lfit(  # noqa: PLR0915
    adata: ad.AnnData,
    between_column: str,
    control_condition: str,
    covariate_column: str | None = None,
    control_max_missing: int = 4,
) -> dict:
    """Perform a linear fit on the data in adata while dealing with NaN values.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.
    between_column : str
        Column name in adata.obs representing the experimental conditions.
    control_condition : str
        The name of the control condition in the between_column.
    covariate_column : str | None, optional
        Column name in adata.obs representing covariates, by default None.
    control_max_missing : int
        Tolerance for missing DMSO values: a feature is skipped if
        n_dmso_not_na < n_dmso - control_max_missing.

    Returns
    -------
    dict
        A dictionary containing the results of the linear fit:
        - 'B': Coefficients of the linear model, shape (n_conditions, n_features)
        - 'sigma2': Estimated variance of the residuals, shape (n_features,)
        - 'dfs': Degrees of freedom for each feature, shape (n_features,)
        - 'M_all': Unscaled covariance of the coefficients, shape (n_conditions, n_conditions, n_features)
    """
    # Generate the design matrix and infer data shape
    dm, col_info = build_design_matrix(adata, between_column, covariate_column)

    # Get general shape information
    K = dm.shape[1]
    P = adata.n_vars

    # Mask for extracting control samples
    control_mask = adata.obs[between_column] == control_condition
    control_idxs = np.where(control_mask)[0]
    n_controls = int(control_mask.sum())

    # Convert design matrix and response to numpy arrays
    X = dm.to_numpy()
    Y = adata.X

    # Initialize output arrays
    B = np.full((K, P), np.nan)  # linear fit coefficients
    sigma2 = np.full(P, np.nan)  # variance of residuals
    dfs = np.full(P, np.nan)  # degrees of freedom for each feature
    M_all = np.full((P, K, K), np.nan)  # unscaled covariance of coefficients for each feature

    # Iterate over each feature and perform linear fit
    for j in tqdm(range(P), desc="Fitting features"):
        # Extract the response vector for the current feature
        y = Y[:, j]

        # Extract control values and check missingness; if too many missing, skip this feature
        control_values = y[control_idxs]
        n_control_not_na = np.sum(~np.isnan(control_values))
        if n_control_not_na < n_controls - control_max_missing:
            B[:, j] = np.nan
            sigma2[j] = np.nan
            dfs[j] = np.nan
            M_all[j] = np.nan
            continue

        # Subset to observed values
        obs_mask = ~np.isnan(y)
        y_obs = y[obs_mask]
        X_obs = X[obs_mask, :]

        # Drop those columns from X_obs which only have zeros after subsetting to obs
        live_col_mask = X_obs.sum(axis=0) > 0
        X_live = X_obs[:, live_col_mask]

        # X_live shape check
        if X_live.shape[1] == 0:
            B[:, j] = np.nan
            sigma2[j] = np.nan
            dfs[j] = np.nan
            M_all[j] = np.nan
            continue

        # Save live indices for later scattering back to full shape
        live_col_idxs = np.where(live_col_mask)[0]

        # Fit linear model using generic least squares solver
        B_live, _residuals, rank, _s = np.linalg.lstsq(X_live, y_obs, rcond=None)

        # Degrees of freedom based on current non-missing observations and design matrix rank
        m = y_obs.shape[0]
        df = m - rank

        # DF check: if df <= 0, skip this feature
        if df <= 0:
            B[:, j] = np.nan
            sigma2[j] = np.nan
            dfs[j] = np.nan
            M_all[j] = np.nan
            continue

        # Compute residuals i.e. the difference between observed and fitted values
        resid = y_obs - X_live @ B_live

        # Compute squared residuals and estimate variance of residuals (sigma^2) using the degrees of freedom
        s2 = (resid**2).sum() / df

        # Compute the unscaled covariance matrix without scaling by s2
        M_live = np.linalg.pinv(X_live.T @ X_live)

        # Scatter B_live and M_live back to full shape since some conditions may have been dropped due to missing values
        B_full = np.full(K, np.nan)
        B_full[live_col_idxs] = B_live

        M_full = np.full((K, K), np.nan)
        M_full[np.ix_(live_col_idxs, live_col_idxs)] = M_live

        # Store results in output arrays
        B[:, j] = B_full
        sigma2[j] = s2
        dfs[j] = df
        M_all[j] = M_full

    return {"B": B, "sigma2": sigma2, "dfs": dfs, "M_all": M_all, "design": dm, "col_info": col_info}


def make_contrasts(
    adata: ad.AnnData,
    between_column: str,
    control_condition: str,
    control_is: int = 1,
) -> pd.DataFrame:
    """Generate a contrast matrix for comparing each condition against one control condition

    The number of conditions is denoted as K, the number of comparisons is K-1.

    Parameters
    ----------
    adata : ad.AnnData
        input data, from whose obs the conditions will be extracted
    between_column : str
        Column name in adata.obs representing the experimental conditions.
    control_condition : str
        The name of the control condition in the between_column.
    control_is : int
        What the control is in the contrast. If it is 1, the treatment is -1 and the effective fold change is control - treatment. If it is -1, the treatment is 1 and the effective fold change is treatment - control.

    Returns
    -------
    pd.DataFrame
        Contrast matrix of shape (K-1, K) where each row represents a comparison of one condition against the control condition. Depending on the value of control_is, the control condition is either 1 or -1 in the contrast, and the treatment condition is the opposite. All other conditions are 0.

    """
    if between_column not in adata.obs.columns:
        raise ValueError(f"Condition column '{between_column}' not found in adata.obs.")

    condition_names = list(adata.obs[between_column].unique())
    if control_condition not in condition_names:
        raise ValueError(f"Control condition '{control_condition}' not found in condition column '{between_column}'.")

    # Treatments are all conditions other than the control; their order defines the row order of the matrix
    treatment_names = [c for c in condition_names if c != control_condition]
    control_idx = condition_names.index(control_condition)

    # Build the (K-1, K) matrix: control column is control_is everywhere, each row's own treatment column is -control_is
    contrast_matrix = np.zeros((len(treatment_names), len(condition_names)), dtype=int)
    contrast_matrix[:, control_idx] = control_is
    for i, treatment in enumerate(treatment_names):
        contrast_matrix[i, condition_names.index(treatment)] = -control_is

    return pd.DataFrame(contrast_matrix, columns=condition_names)


def run_contrasts(
    contrast_matrix: pd.DataFrame,
    B: np.ndarray,  # noqa: N803
    M_all: np.ndarray,  # noqa: N803
    col_info: dict,
) -> dict:
    """Compute Log2 fold changes, unscaled variances, and unscaled standard deviations for each precursor based on the contrast matrix.

    Parameters
    ----------
    contrast_matrix : pd.DataFrame
        A DataFrame representing the contrast matrix, where rows are contrasts and columns are conditions.
    B : np.ndarray
        Coefficients of the linear model, shape (n_conditions, n_features).
    M_all : np.ndarray
        Unscaled covariance of the coefficients, shape (n_features, n_conditions, n_conditions).
    col_info : dict
        A dictionary containing information about the design matrix columns, including 'condition_col_idxs' which maps condition names to their corresponding column indices in B and M_all.

    Returns
    -------
    dict
        A dictionary containing:
        - 'log2fc': Log2 fold changes for each contrast and feature, shape (n_contrasts, n_features).
        - 'unscaled_var': Unscaled variances for each contrast and feature, shape (n_contrasts, n_features).
        - 'stdev_unscaled': Unscaled standard deviations for each contrast and feature, shape (n_contrasts, n_features).

    """
    # Resolve contrast_matrix columns (condition names) to row indices in B / M_all
    condition_col_idxs = col_info["condition_col_idxs"]
    cond_idxs = np.array([condition_col_idxs[name] for name in contrast_matrix.columns])

    # Subset B and M_all to the condition rows / cols (drops covariate rows)
    B_cond = B[cond_idxs]  # (n_conditions, P)
    M_cond = M_all[:, cond_idxs, :][:, :, cond_idxs]  # (P, n_conditions, n_conditions)

    C = contrast_matrix.to_numpy()  # (n_contrasts, n_conditions)
    log2fc = C @ B_cond  # (n_contrasts, P)

    # unscaled variance: per-precursor quadratic form C[c] @ M_cond[j] @ C[c].
    # einsum collapses the (feature, contrast) loops into one vectorized call.
    # NaN propagation matches the explicit loop: 0 * nan = nan, so any contrast
    # touching a dropped condition column still yields nan.
    unscaled_var = np.einsum("ca,jab,cb->cj", C, M_cond, C)  # (n_contrasts, P)

    stdev_unscaled = np.sqrt(unscaled_var)  # (n_contrasts, P)
    return {"log2fc": log2fc, "unscaled_var": unscaled_var, "stdev_unscaled": stdev_unscaled}


def contrasts_from_matrix(
    contrast_matrix: pd.DataFrame,
    control_condition: str,
) -> list[str]:
    """Generate a list of contrast names from the contrast matrix.

    The name encodes the sign of the computed effect, where "A_VS_B" denotes A - B. Since
    log2fc = C @ B, a control of 1 and treatment of -1 yields control - treatment (named
    "control_VS_treatment"), while a control of -1 and treatment of 1 yields treatment - control
    (named "treatment_VS_control").

    Parameters
    ----------
    contrast_matrix : pd.DataFrame
        A DataFrame representing the contrast matrix.
    control_condition : str
        The name of the control condition.

    Returns
    -------
    list[str]
        A list of contrast names.

    """
    if control_condition not in contrast_matrix.columns:
        raise ValueError(f"Control condition '{control_condition}' not found in contrast matrix columns.")

    if list(contrast_matrix.columns).count(control_condition) > 1:
        raise ValueError(f"Control condition '{control_condition}' occurs more than once in contrast matrix columns.")

    # Each row is one contrast: the control column carries control_is, and exactly one treatment
    # column carries -control_is (all other treatment columns are zero on that row).
    contrast_names = []
    for _, row in contrast_matrix.iterrows():
        control_value = row[control_condition]
        treatment_cols = [col for col in contrast_matrix.columns if col != control_condition and row[col] != 0]
        if len(treatment_cols) != 1:
            raise ValueError(
                f"Each contrast row must have exactly one non-zero treatment column, found {treatment_cols}."
            )
        treatment = treatment_cols[0]
        treatment_value = row[treatment]

        # Name must match the sign of log2fc = C @ B: control=1/treatment=-1 gives
        # control - treatment, control=-1/treatment=1 gives treatment - control.
        if control_value == 1 and treatment_value == -1:
            contrast_names.append(f"{control_condition}_VS_{treatment}")
        elif control_value == -1 and treatment_value == 1:
            contrast_names.append(f"{treatment}_VS_{control_condition}")
        else:
            raise ValueError(
                f"Contrast between '{treatment}' and '{control_condition}' is not valid. Expected one to be 1 and the other to be -1."
            )

    return contrast_names


def ebayes_moderation(
    log2fcs: np.ndarray,
    stdevs_unscaled: np.ndarray,
    sigma2: np.ndarray,
    dfs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Moderate residual variances via empirical Bayes (inmoose limma port).

    Features that were never fit (NaN sigma2/dfs, or df<=0) are excluded from the
    prior estimation and returned as NaN. Within-feature NaNs (inestimable single
    contrasts from dead compound columns) are left to propagate through eBayes.

    Parameters
    ----------
    log2fcs : np.ndarray
        Log2 fold changes for each contrast and feature, shape (n_contrasts, n_features).
    stdevs_unscaled : np.ndarray
        Unscaled standard deviations for each contrast and feature, shape (n_contrasts, n_features).
    sigma2 : np.ndarray
        Estimated residual variances for each feature, shape (n_features,).
    dfs : np.ndarray
        Degrees of freedom for each feature, shape (n_features,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        p-values and moderated t-statistics for each contrast and feature, both of shape (n_contrasts, n_features). Features that were never fit are returned as NaN.

    """
    n_contrasts, P = log2fcs.shape

    # Remove features that have no fit at all, i.e. NaN columns in log2fcs, stdevs_unscaled, or nan in sigma2, or dfs
    valid = np.isfinite(sigma2) & np.isfinite(dfs) & (dfs > 0)

    # inmoose's eBayes asserts `.index` on coefficients/stdev_unscaled (so wrap in DataFrame),
    # but does `df[:, None]` on df_residual internally — keep sigma/df_residual as ndarrays.
    # TODO: swap valid_idx for adata.var_names later so results round-trip back to precursor IDs.
    valid_idx = np.where(valid)[0]
    fit = limma.MArrayLM(
        coefficients=pd.DataFrame(log2fcs[:, valid].T, index=valid_idx),
        stdev_unscaled=pd.DataFrame(stdevs_unscaled[:, valid].T, index=valid_idx),
        sigma=np.sqrt(sigma2[valid]),  # SD, not variance
        df_residual=dfs[valid],
        cov_coef=None,
    )
    fit = limma.eBayes(fit)

    # scatter back to full (88, P), excluded features stay NaN
    p = np.full((n_contrasts, P), np.nan)
    t = np.full((n_contrasts, P), np.nan)
    p[:, valid] = np.asarray(fit.p_value).T
    t[:, valid] = np.asarray(fit.t).T

    return {"p": p, "t": t}


def diff_exp_ebayes(
    adata: ad.AnnData,
    between_column: str,
    comparison: tuple[str | list[str], str],
    covariate_column: str | None = None,
    control_max_missing: int = 0,
    control_is: int = -1,
) -> pd.DataFrame:
    """Run Limma eBayes moderated ttest for differential expression with multiple contrasts and covariate support.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with expression data in .X and sample metadata in .obs.
    between_column : str
        Column name in adata.obs containing the contrast levels.
    comparison : tuple[str | list[str], str]
        Tuple specifying the pair of conditions to compare, e.g. ("treatment1", "control"). Multiple treatment conditions can be specified as a list in the first element: (["treatment1", "treatment2"], "control"). If the first element is set to "_ALL_", all conditions except the control will be compared against the control: ("_ALL_", "control").
    covariate_column : str | None, optional
        Column name in adata.obs containing linear covariate levels, by default None.
    control_max_missing : int, optional
        Tolerance for missing values in the control condition. Features with more than this number of missing values in the control condition will be skipped, by default 0.
    control_is : int, optional
        Determines how the control is represented in the contrast matrix. If 1, the treatment is -1 and the effective fold change is control - treatment; if -1, the treatment is 1 and the effective fold change is treatment - control. Default is -1.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized Limma eBayes differential expression results for each contrast.

    """
    if between_column not in adata.obs.columns:
        raise ValueError(f"Column '{between_column}' not found in adata.obs.")
    between_levels = adata.obs[between_column].unique()

    # Validate control condition
    control_condition = comparison[1]
    if control_condition not in between_levels:
        raise ValueError(f"Control condition '{control_condition}' not found in column '{between_column}'.")

    # Validate treatment conditions
    treatment_conditions = comparison[0]
    if treatment_conditions == "_ALL_":
        treatment_conditions = [level for level in between_levels if level != control_condition]
    elif isinstance(treatment_conditions, str):
        treatment_conditions = [treatment_conditions]

    for treatment in treatment_conditions:
        if treatment not in between_levels:
            raise ValueError(f"Treatment condition '{treatment}' not found in column '{between_column}'.")

    # Filter adata to only include samples from the specified conditions
    selected_levels = [*treatment_conditions, control_condition]
    adata = adata[adata.obs[between_column].isin(selected_levels)].copy()

    # Step 1: linear fit with NaN handling
    lm_fit = nan_lfit(
        adata=adata,
        between_column=between_column,
        control_condition=control_condition,
        covariate_column=covariate_column,
        control_max_missing=control_max_missing,
    )

    # Step 2: Generate contrasts to derive fold changes for each treatment vs control
    contrast_matrix = make_contrasts(
        adata=adata,
        between_column=between_column,
        control_condition=control_condition,
        control_is=control_is,
    )

    # Step 3: Run contrasts to compute log2 fold changes and unscaled variances
    contrast_results = run_contrasts(
        contrast_matrix=contrast_matrix,
        B=lm_fit["B"],
        M_all=lm_fit["M_all"],
        col_info=lm_fit["col_info"],
    )

    # Step 4: Run empirical Bayes moderation on the unscaled variances
    ebayes_results = ebayes_moderation(
        log2fcs=contrast_results["log2fc"],
        stdevs_unscaled=contrast_results["stdev_unscaled"],
        sigma2=lm_fit["sigma2"],
        dfs=lm_fit["dfs"],
    )

    # Assemble results into an output dataframe, for multiple contrasts, return a dict of contrasts with their respective results.

    contrast_names = contrasts_from_matrix(contrast_matrix, control_condition)
    if len(contrast_names) != contrast_results["log2fc"].shape[0]:
        raise ValueError("Number of contrast names does not match number of contrasts in results.")

    results = {}
    for contrast_idx, contrast_name in enumerate(contrast_names):
        # preprocess p-values and FDR for the current contrast
        fdr_pvalues = nan_safe_bh_correction(ebayes_results["p"][contrast_idx])
        neg_log10_fdr = np.array([negative_log10_pvalue(fdr) for fdr in fdr_pvalues])
        p_values = ebayes_results["p"][contrast_idx]
        neg_log10_pvalues = np.array([negative_log10_pvalue(p) for p in p_values])

        # Sample counts per level, mirroring ebayes.py. The name is "level_1_VS_level_2".
        level_1, level_2 = contrast_name.split("_VS_")
        max_level_1_samples, max_level_2_samples = determine_max_replicates(adata, between_column, level_1, level_2)

        results[contrast_name] = pd.DataFrame(
            {
                "condition_pair": contrast_name,
                "protein": adata.var_names,
                "log2fc": contrast_results["log2fc"][contrast_idx],
                "p_value": p_values,
                "-log10(p_value)": neg_log10_pvalues,
                "fdr": fdr_pvalues,
                "-log10(fdr)": neg_log10_fdr,
                "method": "limma_ebayes_inmoose_expanded",
                "max_level_1_samples": max_level_1_samples,
                "max_level_2_samples": max_level_2_samples,
            },
            index=adata.var_names,
        )

    return results
