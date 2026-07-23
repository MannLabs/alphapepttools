from typing import Literal

import numpy as np
import pandas as pd


def _is_data_complete(
    data: np.ndarray,
) -> bool:
    """Check if data contains any missing values

    Parameters
    ----------
    data
        Samples x Features array

    Returns
    -------
    bool
        True if data contains no missing values, False otherwise
    """
    return not np.any(np.isnan(data))


def _raise_on_nan_values(
    data: np.ndarray | pd.DataFrame | pd.Series,
    mode: Literal["any", "all"] = "any",
    custom_message: str | None = None,
) -> None:
    """Check if data contains nan values

    Toggle the mode to raise on any or all nan values. If checking for any nan values, columns are checked in the case of
    DataFrames/Series. If checking for all nan values, the entire DataFrame/Series/array is checked.

    Parameters
    ----------
    data
        Samples x Features array
    mode
        "any": Raise if any nan value is present
        "all": Raise if all values are nans

    Raises
    ------
    ValueError
        If data contains nan values based on the specified mode

    """
    if mode == "any":
        has_nans = pd.isna(data).any().any() if isinstance(data, (pd.DataFrame, pd.Series)) else np.isnan(data).any()
        if has_nans:
            raise ValueError(f"Data contains nan values. {custom_message or ''}")
    elif mode == "all":
        if isinstance(data, (pd.DataFrame, pd.Series)):
            all_nan_columns = pd.isna(data).all()
            if any(all_nan_columns):
                raise ValueError(
                    f"Columns with index {all_nan_columns.index.tolist()} contain all nan values. {custom_message or ''}"
                )
        else:
            all_nan_features = np.isnan(data).all(axis=0)
            if any(all_nan_features):
                raise ValueError(
                    f"Features with index {(np.where(all_nan_features)[0]).tolist()} contain all nan values. {custom_message or ''}"
                )
    else:
        raise ValueError("Mode must be either 'any' or 'all'.")
