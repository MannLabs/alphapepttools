# Statistics functionalities for working with AnnData objects

import logging

import numpy as np
from scipy.stats import ttest_ind

# logging configuration
logging.basicConfig(level=logging.INFO)


# Independent ttest for two groups with additional handling of incomplete data
def nan_safe_ttest_ind(
    a: np.ndarray,
    b: np.ndarray,
    min_group_size: int = 2,
    **kwargs,
) -> tuple:
    """Perform an independent t-test on two groups.

    Additionally, handle incomplete input data (where one of the groups has
    fewer than 2 non-NA values) by returning (np.nan, np.nan). While the ttest_ind
    computation runs with a single value in one group the result should not be used
    because the test statistic is based only on the variance from one group.

    Parameters
    ----------
    a : np.ndarray
        First group for t-test
    b : np.ndarray
        Second group for t-test
    min_group_size : int
        Minimum number of non-NA values required in each group to perform the t-test
    **kwargs: dict
        Additional keyword arguments for scipy.stats.ttest_ind, such as equal_var and nan_policy

    Returns
    -------
    tuple : TtestResult (tuple) or tuple of np.nan
        Tuple of t-statistic and p-value for the t-test. The first alue is the statistic, the
        second value is the two-tailed p-value.

    """
    # Ensure proper type for input data
    a = np.array(a)
    b = np.array(b)

    if (~np.isnan(a)).sum() < min_group_size or (~np.isnan(b)).sum() < min_group_size:
        return (np.nan, np.nan)
    return ttest_ind(
        a[~np.isnan(a)],
        b[~np.isnan(b)],
        **kwargs,
    )


def pca() -> None:
    raise NotImplementedError


# Automatically define __all__ to contain public names
__all__: list[str] = [name for name in globals() if not name.startswith("_")]
