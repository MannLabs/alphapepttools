# Statistics functionalities for working with AnnData objects

import logging
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ttest_ind

# logging configuration
logging.basicConfig(level=logging.INFO)


def nan_safe_ttest_ind(  # implicitly tested via unit-test group_ratios_ttest_ind
    a: pd.Series,
    b: pd.Series,
    **kwargs,
) -> tuple[float, float]:
    """Perform t-test between two pandas Series, handling NaNs safely.

    Parameters
    ----------
    a : pd.Series
        First series of data.
    b : pd.Series
        Second series of data.
    **kwargs : dict
        Additional keyword arguments to pass to `scipy.stats.ttest_ind`.

    Returns
    -------
    tuple[float, float]

    """
    MIN_REQUIRED_SAMPLES = 2

    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        warnings.warn(
            " --> nan_safe_ttest_ind warning: Input must be a pandas Series. Converting to series...",
            stacklevel=2,
        )
        a = pd.Series(a)
        b = pd.Series(b)

    if a.count() < MIN_REQUIRED_SAMPLES or b.count() < MIN_REQUIRED_SAMPLES:
        return (np.nan, np.nan)
    return ttest_ind(a, b, **kwargs)


def nan_safe_bh_correction(  # implicitly tested via unit-test group_ratios_ttest_ind
    pvals: np.array,
) -> np.array:
    """Scipy.stats.false_discovery_control with nan-safe handling

    Scipy.stats.false_discovery_control is not nan-safe, we need to delete nans, apply correction, then re-insert nans.
    This method adds a unique index to the input array, drops nans, applies correction, then merges the outcome back to
    the original array indices.

    Parameters
    ----------
        pvals (np.array): Array of p-values.
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


def ttest(
    adata: ad.AnnData,
    between: str,
    *,
    equal_var: bool = False,
) -> pd.DataFrame:
    """Calculate (log2) ratios of features between metadata groups.

    Calculate ratios and log2 ratios of each feature in the input data
    between groups in the metadata 'between' column.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing data and metadata.
    between : str
        Name of the column in sample metadata that contains the groups
        to compare.
    equal_var : bool
        Whether to assume equal variance in the t-test.

    Returns
    -------
    pd.DataFrame
        Dataframe with ratios and log2 ratios, as well as p-values
        of each feature between the groups in the 'between' column.

    """
    MIN_REQUIRED_SAMPLES = 2

    # Parse out dataframes
    data = adata.to_df()
    metadata = adata.obs

    # check that all columns of data are numeric
    if not all(np.issubdtype(data[col].dtype, np.number) for col in data.columns):
        logging.warning("Error in group_ratios_ttest_ind: All columns of data must be numeric.")
        return None

    # check input groups
    levels = metadata[between].unique()

    # ratios require at least two unique groups
    if len(levels) != MIN_REQUIRED_SAMPLES:
        logging.warning("Error in group_ratios_ttest_ind: 'between' variable must have exactly two levels.")
        return None

    # each group must have at least two non-na samples
    if any(metadata[between].value_counts() < MIN_REQUIRED_SAMPLES):
        logging.warning(
            "Error in group_ratios_ttest_ind: Each group in 'between' variable must have at least two samples."
        )
        return None

    # generate all possible ratio pairs and inverses between groups
    _out = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            g1 = levels[i]
            g2 = levels[j]

            g1_frame = data[metadata[between] == g1]
            g2_frame = data[metadata[between] == g2]

            comparison = f"{g1}_over_{g2}"
            inverted_comparison = f"{g2}_over_{g1}"
            features = pd.Series(data.columns)

            # record number of non-na samples in each group
            g1_n_samples = g1_frame.count()
            g2_n_samples = g2_frame.count()

            # calculate ratio and difference, latter for log-transformed inputs
            g1_mean = g1_frame.mean(axis=0)
            g2_mean = g2_frame.mean(axis=0)

            # where mean is zero, insert np.nan to avoid division by zero
            g1_mean[g1_mean == 0] = np.nan
            g2_mean[g2_mean == 0] = np.nan

            # calculate ratios and deltas
            ratio = g1_mean / g2_mean
            delta = g1_mean - g2_mean

            # apply wrapped ttest_ind that returns nan if either group
            # has fewer than two non-na samples
            t, p = zip(
                *features.apply(
                    lambda x, _g1_frame=g1_frame, _g2_frame=g2_frame: nan_safe_ttest_ind(
                        a=_g1_frame[x],
                        b=_g2_frame[x],
                        equal_var=equal_var,
                        nan_policy="omit",
                    )
                ),
                strict=False,
            )

            # adjust pvalues using Benjamini-Hochberg method, accounting for nans
            p_adj = nan_safe_bh_correction(p)

            # store results in dataframe
            _out_df = pd.DataFrame(
                {
                    "id": features.to_numpy(),
                    f"ratio_{comparison}": ratio,
                    f"delta_{comparison}": delta,
                    f"tvalue_{comparison}": t,
                    f"pvalue_{comparison}": p,
                    f"padj_{comparison}": p_adj,
                    f"ratio_{inverted_comparison}": 1 / ratio,
                    f"delta_{inverted_comparison}": -delta,
                    f"tvalue_{inverted_comparison}": [-tj for tj in t],
                    f"pvalue_{inverted_comparison}": p,
                    f"padj_{inverted_comparison}": p_adj,
                    f"n_{g1}": g1_n_samples,
                    f"n_{g2}": g2_n_samples,
                }
            ).set_index("id", drop=True)
            _out_df.index.name = None

            _out.append(_out_df)

    return pd.concat(_out, axis=1)


def pca() -> None:
    """Perform a PCA on the data"""
    raise NotImplementedError
