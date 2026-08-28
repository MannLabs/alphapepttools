# Expanded eBayes differential expression module with multiple contrasts, nan-handling and covariate support

from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from inmoose import limma

    _HAS_INMOOSE = True
except ModuleNotFoundError:
    _HAS_INMOOSE = False

from alphapepttools._utils import get_matrix
from alphapepttools.tl.defaults import tl_defaults
from alphapepttools.tl.stats import nan_safe_bh_correction
from alphapepttools.tl.utils import (
    determine_max_replicates,
    negative_log10_pvalue,
)

_METHOD_NAME = "limma_ebayes_inmoose_expanded"


def _one_hot(labels: pd.Series, *, drop_first: bool = False) -> pd.DataFrame:
    """One-hot encode sample labels, keeping columns in order of first appearance.

    Casting to ``object`` first is what makes this safe for AnnData: ``pd.get_dummies`` on a
    categorical column emits one column per *declared* category, so a level that no longer
    occurs (e.g. after subsetting an AnnData) would contribute an all-zero, rank-deficient
    column to the design matrix. Casting to ``object`` rather than ``str`` keeps the original
    label values, which downstream lookups (e.g. contrast matrix columns) are keyed on.

    Parameters
    ----------
    labels : pd.Series
        Sample labels to encode, indexed by sample.
    drop_first : bool, optional
        If True, drop the first level to avoid multicollinearity (k-1 encoding). "First" is in
        the ``pd.get_dummies`` sense, i.e. the lexicographically first level, not the first to
        appear in the data.

    Returns
    -------
    pd.DataFrame
        Indicator matrix indexed like ``labels``, with columns ordered by first appearance.

    """
    labels = labels.astype(object)
    dm = pd.get_dummies(labels, dtype=int, drop_first=drop_first)
    # get_dummies sorts its columns; restore order of first appearance, minus any dropped level.
    return dm[[level for level in dict.fromkeys(labels) if level in dm.columns]]


def _build_design_matrix(
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
        A design matrix suitable for linear modeling, where columns are features and rows are samples. Each condition is encoded as a separate column. Covariate levels are added as additional columns in a k-1 fashion (one-hot encoding with the first level dropped to avoid multicollinearity). Within each block, columns follow the order the levels first appear in adata.obs.
    dict
        A dictionary to navigate the design matrix with the following keys:
        - condition_col_idxs: A dictionary mapping each condition to its corresponding column index in the design matrix.
        - covariate_col_idxs: A dictionary mapping each covariate level to its corresponding column index in the design matrix. Empty if covariate_column is None.

    """
    if condition_column not in adata.obs.columns:
        raise KeyError(f"Condition column '{condition_column}' not found in adata.obs.")

    if adata.obs[condition_column].isna().any():
        raise KeyError(f"Condition column '{condition_column}' contains NaN values.")

    if covariate_column is not None:
        if covariate_column not in adata.obs.columns:
            raise KeyError(f"Covariate column '{covariate_column}' not found in adata.obs.")
        if adata.obs[covariate_column].isna().any():
            raise KeyError(f"Covariate column '{covariate_column}' contains NaN values.")

    condition_dm = _one_hot(cast("pd.Series", adata.obs[condition_column]))
    covariate_dm = (
        _one_hot(cast("pd.Series", adata.obs[covariate_column]), drop_first=True)  # k-1 for covariates
        if covariate_column is not None
        else pd.DataFrame(index=adata.obs.index)
    )

    dm = pd.concat([condition_dm, covariate_dm], axis=1)

    condition_col_idxs = {name: dm.columns.get_loc(name) for name in condition_dm.columns}
    covariate_col_idxs = {name: dm.columns.get_loc(name) for name in covariate_dm.columns}

    return dm, {"condition_col_idxs": condition_col_idxs, "covariate_col_idxs": covariate_col_idxs}


def print_design_matrix_summary(
    dm: pd.DataFrame,
) -> None:
    """Summarize the design matrix by reporting the counts of samples for each condition and covariate level."""
    counts = dm.sum(axis=0)
    columns = dm.columns.tolist()
    order = np.argsort(counts.values)[::-1]
    print("Design Matrix Summary:")
    for col in np.array(columns)[order]:
        print(f"{col}: {counts[col]}")


def _nan_lmfit(
    adata: ad.AnnData,
    design_matrix: pd.DataFrame,
) -> dict:
    """Perform a linear fit on the data in adata while dealing with NaN values.

    Fits every feature; pre-fit completeness filtering, if wanted, is the caller's responsibility
    (e.g. via alphapepttools.pp.filter_data_completeness).

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.
    design_matrix : pd.DataFrame
        Design matrix with samples as rows (aligned to adata.obs_names) and conditions/covariates
        as columns, e.g. as produced by build_design_matrix.

    Returns
    -------
    dict
        A dictionary containing the results of the linear fit:
        - 'B': Coefficients of the linear model, shape (n_conditions, n_features)
        - 'sigma2': Estimated variance of the residuals, shape (n_features,)
        - 'dfs': Degrees of freedom for each feature, shape (n_features,)
        - 'M_all': Unscaled covariance of the coefficients, shape (n_conditions, n_conditions, n_features)
    """
    # Validate that the design matrix aligns with the adata samples
    if design_matrix.shape[0] != adata.n_obs:
        raise ValueError("Design matrix rows do not match the number of samples in adata.")
    if not design_matrix.index.equals(adata.obs_names):
        raise ValueError("Design matrix index does not align with adata.obs_names.")

    # Get general shape information
    K = design_matrix.shape[1]
    P = adata.n_vars

    # Convert design matrix and response to numpy arrays
    X = design_matrix.to_numpy()
    Y = get_matrix(adata)

    # Initialize output arrays
    B = np.full((K, P), np.nan)  # linear fit coefficients
    sigma2 = np.full(P, np.nan)  # variance of residuals
    dfs = np.full(P, np.nan)  # degrees of freedom for each feature
    M_all = np.full((P, K, K), np.nan)  # unscaled covariance of coefficients for each feature

    # Iterate over each feature and perform linear fit
    for j in tqdm(range(P), desc="Fitting features"):
        # Extract the response vector for the current feature
        y = Y[:, j]

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

        # Skip features with insufficient degrees of freedom
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

        # Store results in output arrays with correct shapes
        B[:, j] = B_full
        sigma2[j] = s2
        dfs[j] = df
        M_all[j] = M_full

    return {"B": B, "sigma2": sigma2, "dfs": dfs, "M_all": M_all}


def _make_contrasts(
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
        raise KeyError(f"Condition column '{between_column}' not found in adata.obs.")

    condition_names = list(adata.obs[between_column].unique())
    if control_condition not in condition_names:
        raise KeyError(f"Control condition '{control_condition}' not found in condition column '{between_column}'.")

    # Treatments are all conditions other than the control; their order defines the row order of the matrix
    treatment_names = [c for c in condition_names if c != control_condition]
    control_idx = condition_names.index(control_condition)

    # Build the (K-1, K) matrix: control column is control_is everywhere, each row's own treatment column is -control_is
    contrast_matrix = np.zeros((len(treatment_names), len(condition_names)), dtype=int)
    contrast_matrix[:, control_idx] = control_is
    for i, treatment in enumerate(treatment_names):
        contrast_matrix[i, condition_names.index(treatment)] = -control_is

    # pd.DataFrame does not take a plain list of labels for `columns` (pandas' Axes type covers
    # Index/ndarray, not builtin list); pd.Index keeps the plain-object column labels.
    return pd.DataFrame(contrast_matrix, columns=pd.Index(condition_names))


def _run_contrasts(
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


def _contrasts_from_matrix(
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

    Raises
    ------
    KeyError
        If `control_condition` is not one of the contrast matrix columns.
    ValueError
        If the contrast matrix is malformed: the control column occurs more than once, or a row does
        not carry exactly one treatment column with the sign opposite to the control.

    """
    if control_condition not in contrast_matrix.columns:
        raise KeyError(f"Control condition '{control_condition}' not found in contrast matrix columns.")

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
            contrast_names.append(f"{control_condition}{tl_defaults.CONDITION_SPLIT_STRING}{treatment}")
        elif control_value == -1 and treatment_value == 1:
            contrast_names.append(f"{treatment}{tl_defaults.CONDITION_SPLIT_STRING}{control_condition}")
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
) -> dict[str, np.ndarray]:
    """Moderate residual variances via empirical Bayes (inmoose limma port).

    Features that were never fit (NaN sigma2/dfs, or df<=0) are excluded from the
    prior estimation and returned as NaN. Within-feature NaNs (inestimable single
    contrasts from dead coefficients columns) are left to propagate through eBayes.

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
    dict[str, np.ndarray]
        A dictionary containing:
        - 'p': p-values for each contrast and feature, shape (n_contrasts, n_features).
        - 't': moderated t-statistics for each contrast and feature, same shape.
        Features that were never fit are returned as NaN.

    """
    n_contrasts, P = log2fcs.shape

    # Remove features that have no fit at all, i.e. NaN columns in log2fcs, stdevs_unscaled, or nan in sigma2, or dfs
    valid_mask = np.isfinite(sigma2) & np.isfinite(dfs) & (dfs > 0)

    # inmoose's eBayes asserts `.index` on coefficients/stdev_unscaled (so wrap in DataFrame),
    # but does `df[:, None]` on df_residual internally — keep sigma/df_residual as ndarrays.
    # TODO: swap valid_idx for adata.var_names later so results round-trip back to precursor IDs.
    valid_idx = np.where(valid_mask)[0]
    # TODO: get rid of limma.MArrayLM once a port is in place; this is only a container to call eBayes with the expected input
    fit = limma.MArrayLM(
        coefficients=pd.DataFrame(log2fcs[:, valid_mask].T, index=valid_idx),
        stdev_unscaled=pd.DataFrame(stdevs_unscaled[:, valid_mask].T, index=valid_idx),
        sigma=np.sqrt(sigma2[valid_mask]),  # SD, not variance
        df_residual=dfs[valid_mask],
        cov_coef=None,
    )
    fit = limma.eBayes(fit)

    # scatter back to full (n_contrasts, P), excluded features stay NaN
    p = np.full((n_contrasts, P), np.nan)
    t = np.full((n_contrasts, P), np.nan)
    p[:, valid_mask] = np.asarray(fit.p_value).T
    t[:, valid_mask] = np.asarray(fit.t).T

    return {"p": p, "t": t}


def _resolve_comparison(
    adata: ad.AnnData,
    between_column: str,
    comparison: tuple[str | list[str], str],
) -> tuple[list[str], str]:
    """Validate `between_column` and resolve `comparison` into explicit A conditions and the B reference.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object whose .obs carries the condition labels.
    between_column : str
        Column name in adata.obs containing the contrast levels.
    comparison : tuple[str | list[str], str]
        Comparison as (A, B), where A is a single condition, a list of conditions, or the "_ALL_"
        sentinel (every level except B), and B is the single reference condition.

    Returns
    -------
    tuple[list[str], str]
        The A conditions as an explicit list (sentinel expanded, single strings wrapped), and the
        B reference condition.

    Raises
    ------
    KeyError
        If `between_column` is absent from adata.obs, or if any resolved condition is not a level of it.

    """
    if between_column not in adata.obs.columns:
        raise KeyError(f"Column '{between_column}' not found in adata.obs.")
    between_levels = adata.obs[between_column].unique()

    # Validate the B condition (the single reference, comparison[1])
    b_condition = comparison[1]
    if b_condition not in between_levels:
        raise KeyError(f"Condition '{b_condition}' not found in column '{between_column}'.")

    # Resolve the A conditions (comparison[0]) to an explicit list
    a_conditions = comparison[0]
    if a_conditions == "_ALL_":
        a_conditions = [level for level in between_levels if level != b_condition]
    elif isinstance(a_conditions, str):
        a_conditions = [a_conditions]

    for a_condition in a_conditions:
        if a_condition not in between_levels:
            raise KeyError(f"Condition '{a_condition}' not found in column '{between_column}'.")

    return list(a_conditions), b_condition


def _sufficient_values_mask(
    adata: ad.AnnData,
    between_column: str,
    condition: str,
    min_required: int,
) -> np.ndarray:
    """Per-feature boolean mask: True where `condition` has at least `min_required` observed values."""
    condition_idxs = np.where(adata.obs[between_column] == condition)[0]
    n_observed = np.sum(~np.isnan(get_matrix(adata)[condition_idxs, :]), axis=0)
    return n_observed >= min_required


def _replicate_gate_mask(
    adata: ad.AnnData,
    between_column: str,
    a_level: str,
    b_level: str,
    a_min_required: int | None,
    b_min_required: int | None,
) -> np.ndarray:
    """Per-feature boolean mask: True where both conditions of a contrast have enough observed values.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object subset to the samples that were fit.
    between_column : str
        Column name in adata.obs containing the contrast levels.
    a_level : str
        The A condition of the contrast.
    b_level : str
        The B condition of the contrast.
    a_min_required : int | None
        Minimum number of observed values required in A. If None, the A gate is disabled.
    b_min_required : int | None
        Minimum number of observed values required in B. If None, the B gate is disabled.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_features,). All True if both gates are disabled.

    """
    keep_mask = np.ones(adata.n_vars, dtype=bool)
    if a_min_required is not None:
        keep_mask &= _sufficient_values_mask(adata, between_column, a_level, a_min_required)
    if b_min_required is not None:
        keep_mask &= _sufficient_values_mask(adata, between_column, b_level, b_min_required)
    return keep_mask


def _standardize_contrast_frame(
    contrast_name: str,
    var_names: pd.Index,
    log2fc: np.ndarray,
    p_values: np.ndarray,
    max_level_1_samples: int,
    max_level_2_samples: int,
) -> pd.DataFrame:
    """Assemble one contrast's results into the shared differential expression output format.

    FDR correction is applied here, so any gating of `p_values` (e.g. the replicate gate) must
    already have been applied by the caller.

    Parameters
    ----------
    contrast_name : str
        Name of the contrast, "A_VS_B" by convention. Written to the condition_pair column.
    var_names : pd.Index
        Feature names, used as both the frame index and the protein column.
    log2fc : np.ndarray
        Log2 fold changes for this contrast, shape (n_features,).
    p_values : np.ndarray
        p-values for this contrast, shape (n_features,).
    max_level_1_samples : int
        Number of samples in the A condition.
    max_level_2_samples : int
        Number of samples in the B condition.

    Returns
    -------
    pd.DataFrame
        Results frame carrying exactly the shared `tl_defaults.DIFF_EXP_COLS` columns, in that order.

    """
    fdr_pvalues = nan_safe_bh_correction(p_values)

    df = pd.DataFrame(
        {
            "condition_pair": contrast_name,
            "protein": var_names,
            "log2fc": log2fc,
            "p_value": p_values,
            "-log10(p_value)": [negative_log10_pvalue(p) for p in p_values],
            "fdr": fdr_pvalues,
            "-log10(fdr)": [negative_log10_pvalue(fdr) for fdr in fdr_pvalues],
            "method": _METHOD_NAME,
            "max_level_1_samples": max_level_1_samples,
            "max_level_2_samples": max_level_2_samples,
        },
        index=var_names,
    )

    # Reindex on the shared column contract so this method stays aligned with the other diff_exp methods
    return df[tl_defaults.DIFF_EXP_COLS]


def diff_exp_ebayes(
    adata: ad.AnnData,
    between_column: str,
    comparison: tuple[str | list[str], str],
    covariate_column: str | None = None,
    a_min_required: int | None = None,
    b_min_required: int | None = None,
    return_coefficients: bool = False,  # noqa: FBT001, FBT002
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Run Limma eBayes moderated ttest for differential expression with multiple contrasts and covariate support.

    The two conditions in each comparison are referred to positionally as A (comparison[0]) and B (comparison[1]);
    the test is symmetric, so no condition is assumed to be a treatment or a control. Missingness handling inside
    this function is limited to gating the reported fold changes: per contrast, a feature's fold change and
    p-value are suppressed (set to NaN, before FDR correction) unless both conditions have at least the required
    number of observed values (a_min_required for A and b_min_required for B). All features are still fit and
    contribute to the eBayes variance prior. Pre-fit completeness filtering, if wanted, is the caller's
    responsibility and should be done upstream (e.g. alphapepttools.pp.filter_data_completeness).

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with expression data in .X and sample metadata in .obs.
    between_column : str
        Column name in adata.obs containing the contrast levels.
    comparison : tuple[str | list[str], str]
        Tuple specifying the pair of conditions to compare, ordered as (A, B): the first element is the A
        condition(s), the second is the single B reference each A is compared against, e.g. ("A", "B"). Fold
        changes are reported as A - B. Multiple A conditions can be specified as a list in the first element:
        (["A1", "A2"], "B"). If the first element is set to "_ALL_", all conditions except B are compared against
        it: ("_ALL_", "B").
    covariate_column : str | None, optional
        Column name in adata.obs containing linear covariate levels, by default None.
    a_min_required : int | None, optional
        Minimum number of observed values required in the A condition (comparison[0]) of each contrast. Per
        contrast, features with fewer observed values in A have their fold change suppressed (set to NaN) before
        FDR correction. If None, the A gate is disabled. By default None.
    b_min_required : int | None, optional
        Minimum number of observed values required in the B condition (comparison[1]). Per contrast, features with
        fewer observed values in B have their fold change suppressed (set to NaN) before FDR correction. If None,
        the B gate is disabled. By default None.
    return_coefficients : bool, optional
        If True, additionally return the full fitted linear coefficients as a (features x design columns) DataFrame,
        with one column per condition and covariate level. Useful for inspecting effect sizes directly, e.g.
        diagnosing an outsized covariate (batch/replicate) effect. By default False.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], pd.DataFrame | None]
        Always a (results, coefficients) tuple. `results` holds the per-contrast standardized Limma eBayes
        results, keyed by contrast name, each carrying the shared `tl_defaults.DIFF_EXP_COLS` columns. Fold
        changes are reported as A - B (i.e. comparison[0] - comparison[1]), and contrasts are named "A_VS_B".
        `coefficients` is the fitted-coefficient DataFrame if `return_coefficients` is True, else None.

    Raises
    ------
    ImportError
        If inmoose is not installed.
    KeyError
        If `between_column` is not in adata.obs, or any condition in `comparison` is not a level of it.

    """
    if not _HAS_INMOOSE:
        raise ImportError(
            "inmoose is required for diff_exp_ebayes(). Install it through pip or install alphapepttools with the 'full'/'full-stable' extra."
        )

    a_conditions, b_condition = _resolve_comparison(adata, between_column, comparison)

    # Step 0: Filter adata to only include samples from the specified conditions
    selected_levels = [*a_conditions, b_condition]
    adata = adata[adata.obs[between_column].isin(selected_levels)].copy()

    # Step 1: build the design matrix and fit every feature with NaN handling
    design_matrix, col_info = _build_design_matrix(adata, between_column, covariate_column)
    lm_fit = _nan_lmfit(adata, design_matrix)

    # Step 2: Generate contrasts to derive fold changes for each A vs B.
    # control_is=-1 fixes the direction to A - B (comparison[0] - comparison[1]), named "A_VS_B".
    contrast_matrix = _make_contrasts(
        adata=adata,
        between_column=between_column,
        control_condition=b_condition,
        control_is=-1,
    )

    # Step 3: Run contrasts to compute log2 fold changes and unscaled variances
    contrast_results = _run_contrasts(
        contrast_matrix=contrast_matrix,
        B=lm_fit["B"],
        M_all=lm_fit["M_all"],
        col_info=col_info,
    )

    # Step 4: Run empirical Bayes moderation on the unscaled variances
    ebayes_results = ebayes_moderation(
        log2fcs=contrast_results["log2fc"],
        stdevs_unscaled=contrast_results["stdev_unscaled"],
        sigma2=lm_fit["sigma2"],
        dfs=lm_fit["dfs"],
    )

    # Step 5: Extract contrasts and write output DataFrame with standardized columns for each contrast
    contrast_names = _contrasts_from_matrix(contrast_matrix, b_condition)
    if len(contrast_names) != contrast_results["log2fc"].shape[0]:
        raise ValueError("Number of contrast names does not match number of contrasts in results.")

    results = {}
    for contrast_idx, contrast_name in enumerate(contrast_names):
        # By convention the contrast is named "A_VS_B" (comparison[0] vs comparison[1]).
        a_level, b_level = contrast_name.split(tl_defaults.CONDITION_SPLIT_STRING)

        p_values = ebayes_results["p"][contrast_idx].copy()
        log2fc = contrast_results["log2fc"][contrast_idx].copy()

        # Replicate gate: suppress the fold change unless both conditions have enough observed values
        keep = _replicate_gate_mask(adata, between_column, a_level, b_level, a_min_required, b_min_required)
        p_values[~keep] = np.nan
        log2fc[~keep] = np.nan

        # Sample counts per level, mirroring ebayes.py. The name is "A_VS_B".
        max_level_1_samples, max_level_2_samples = determine_max_replicates(adata, between_column, a_level, b_level)

        results[contrast_name] = _standardize_contrast_frame(
            contrast_name=contrast_name,
            var_names=adata.var_names,
            log2fc=log2fc,
            p_values=p_values,
            max_level_1_samples=max_level_1_samples,
            max_level_2_samples=max_level_2_samples,
        )

    if return_coefficients:
        # rows of B are design-matrix columns (conditions + covariates), in column order
        coefficients = pd.DataFrame(
            lm_fit["B"].T,
            index=adata.var_names,
            columns=design_matrix.columns,
        )
        return results, coefficients

    return results, None
