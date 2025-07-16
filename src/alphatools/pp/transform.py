# Data transformations


import logging

import anndata as ad
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_data_integrity(
    data: np.ndarray,
    verbosity: int = 0,
) -> np.ndarray:
    """Detect nonstandard data inputs.

    Current nonstandard values include:
    - NaN values
    - Zero values
    - Negative values
    - Positive infinity
    - Negative infinity

    If `warn` is True, log warnings for found nonstandard values.

    Parameters
    ----------
    data : np.ndarray
        Input data to check for nonstandard values.
    verbosity : int, default 0
        If 1, log warnings for nonstandard values found in the data.

    Returns
    -------
    np.ndarray
        A boolean mask indicating the positions of nonstandard values in the data.
        True indicates a nonstandard value.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy.ndarray.")

    data_status = {}

    data_status["nan"] = np.isnan(data)
    data_status["zero"] = data == 0
    data_status["negative"] = data < 0
    data_status["inf"] = data == np.inf
    data_status["negative_inf"] = data == -np.inf

    data_mask = np.zeros_like(data, dtype=bool)
    for parameter, status in data_status.items():
        if np.any(status):
            if verbosity > 0:
                logger.warning(f"Found {sum(status)} {parameter} values in the data.")
            data_mask |= status

    return data_mask


def nanlog(
    data: ad.AnnData | np.ndarray | pd.DataFrame | pd.Series,
    base: int = 2,
    verbosity: int = 0,
) -> np.ndarray | pd.DataFrame | pd.Series | ad.AnnData:
    """Logarithmize a data matrix.

    Return log-transformed data, replacing zeros and other invalid values with np.nan.
    Original data is not modified.

    Current invalid values include:
    - NaN values
    - Zero values
    - Negative values
    - Positive infinity
    - Negative infinity

    Parameters
    ----------
    x : np.array | pd.DataFrame | pd.Series | anndata.AnnData
        Input data; negatives and/or zeros are converted to np.nan
    base : int
        Base of the logarithm. Defaults to 2 (log2).
    verbosity : int, default 0
        If 1, log warnings for invalid values found in the data.

    Returns
    -------
    np.array | pd.DataFrame | pd.Series | anndata.AnnData
        Log-transformed data with invalid values replaced by np.nan.
        The type of the returned data matches the input type.

    """
    if not isinstance(data, np.ndarray | pd.DataFrame | pd.Series | ad.AnnData):
        raise TypeError("Input must be a anndata.AnnData, numpy.ndarray, pandas.DataFrame or pandas.Series.")

    if base in {0, 1} or base < 0:
        raise ValueError("Base cannot be 0 (divide by -Inf) or 1 (divide by 0) or negative (invalid log).")

    data = data.copy()

    def _log_func(
        x: np.ndarray | pd.DataFrame | pd.Series,
        base: float,
    ) -> np.ndarray | pd.DataFrame | pd.Series:
        """Apply logarithm transformation; for base 2 and 10 use dedicated numpy functions"""
        if base == 2:  # NOQA: PLR2004, log2 is standard jargon
            return np.log2(x)
        if base == 10:  # NOQA: PLR2004, log10 is standard jargon
            return np.log10(x)
        return np.log(x) / np.log(base)

    # Handle subtleties with filtering and assignment of different datatypes
    if isinstance(data, ad.AnnData):
        nanmask = check_data_integrity(data.X, verbosity)
        data.X = _log_func(np.where(~nanmask, data.X, np.nan), base)
    elif isinstance(data, pd.DataFrame | pd.Series):
        nanmask = check_data_integrity(data.to_numpy(), verbosity)
        data = _log_func(data.where(~nanmask, np.nan), base)
    else:
        nanmask = check_data_integrity(data, verbosity)
        data = _log_func(np.where(~nanmask, data, np.nan), base)

    return data
