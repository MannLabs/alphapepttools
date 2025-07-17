# Data transformations


import logging

import anndata as ad
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_special_values(
    data: np.ndarray,
    verbosity: int = 0,
) -> np.ndarray:
    """Detect special values such as NaN, zero, negative, and infinite values in the data.

    This function checks for nonstandard values in the input data and returns a boolean mask indicating
    their position. Additionally, it provides a log summary for the number and kind of nonstandard values found.
    This function is useful for upfront checks of data integrity, e.g. prior to transformations or analyses like
    PCA or clustering.

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

    special_values = {}

    special_values["nan"] = np.isnan(data)
    special_values["zero"] = data == 0
    special_values["negative"] = data < 0
    special_values["inf"] = data == np.inf
    special_values["negative_inf"] = data == -np.inf

    special_values_mask = np.zeros_like(data, dtype=bool)
    for parameter, status in special_values.items():
        if np.any(status):
            if verbosity > 0:
                logger.warning(f"Found {sum(status.flatten())} {parameter} values in the data.")
            special_values_mask |= status

    return special_values_mask


def nanlog(
    data: ad.AnnData,
    base: int = 2,
    verbosity: int = 1,
) -> ad.AnnData:
    """Logarithmize a data matrix.

    Apply arbitrary base logarithm transformation to AnnData.X, replacing invalid values with np.nan.
    Similar to the underlying numpy log functions, invalid values are replaced with np.nan, but a more
    detailed summary of which values were replaced is provided.

    Current invalid values include:
    - NaN values
    - Zero values
    - Negative values
    - Positive infinity
    - Negative infinity

    Parameters
    ----------
    x : anndata.AnnData
        Input data; negatives and/or zeros are converted to np.nan
    base : int
        Base of the logarithm. Defaults to 2 (log2).
    verbosity : int, default 1
        If 1, log warnings for invalid values found in the data.

    Returns
    -------
    anndata.AnnData
        Log-transformed data with invalid values replaced by np.nan.

    """
    if not isinstance(data, ad.AnnData):
        raise TypeError("Input must be a anndata.AnnData.")

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
    special_values_mask = detect_special_values(data.X, verbosity)
    data.X = _log_func(np.where(~special_values_mask, data.X, np.nan), base)

    return data
