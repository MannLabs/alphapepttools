# Data transformations


import anndata as ad
import numpy as np
import pandas as pd


def nanlog(
    data: ad.AnnData | np.ndarray | pd.DataFrame | pd.Series,
    log: int = 2,
) -> np.array:
    """Wrapper for nplog() functions.

    Replace zeros and negatives with nan and return either log2 or log10 transformed values.

    Parameters
    ----------
    x : np.array | pd.DataFrame | pd.Series
        Input data; negatives and/or zeros are converted to np.nan
    log : int
        Log-level, currently allowed values are 2 and 10.

    Returns
    -------
    np.array:
        array with log-transformed original values. Zeros
        and negative values are replaced by np.nan.

    """
    LOG_FUNCTIONS = {2: np.log2, 10: np.log10}

    if not isinstance(data, ad.AnnData | pd.DataFrame | pd.Series | np.ndarray):
        raise TypeError("Input must be a anndata.AnnData, numpy.ndarray, pandas.DataFrame or pandas.Series.")

    if log not in LOG_FUNCTIONS:
        raise TypeError(f"'log' must be any of {LOG_FUNCTIONS.keys()}, but got {log}.")

    # Copy for now to avoid inplace modifications TODO: add inplace option?
    data = data.copy()

    # Prevent edgecases that cause downstream processing issues
    def _sanitized_log_inputs(
        data: np.ndarray | pd.DataFrame | pd.Series,
    ) -> np.ndarray | pd.DataFrame | pd.Series:
        """Sanitize inputs for log transformation."""
        nanmask = np.isnan(data)
        nanmask |= data == 0
        nanmask |= data < 0
        nanmask |= data == np.inf
        nanmask |= data == -np.inf

        # Avoid inplace modification and copy warnings
        if isinstance(data, pd.DataFrame | pd.Series):
            return data.where(~nanmask, np.nan)
        if isinstance(data, np.ndarray):
            return np.where(~nanmask, data, np.nan)
        raise TypeError("Unsupported data type for sanitization.")

    if isinstance(data, ad.AnnData):
        data.X = LOG_FUNCTIONS[log](_sanitized_log_inputs(data.X))
        return data
    return LOG_FUNCTIONS[log](_sanitized_log_inputs(data))
