import logging

import numpy as np
from scipy.stats import false_discovery_control

# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def nan_safe_bh_correction(
    pvals: np.array,
) -> np.array:
    """Apply Benjamini-Hochberg correction with NaN-safe handling.

    Scipy.stats.false_discovery_control is not nan-safe, we need to delete nans, apply correction, then re-insert nans.
    This method preserves nans in their original positions while applying BH correction to valid p-values.

    Parameters
    ----------
    pvals : np.array
        Array of p-values, may contain NaNs.

    Returns
    -------
    np.array
        Array with BH-corrected p-values, NaNs preserved in original positions.

    Examples
    --------
    >>> import numpy as np
    >>> from alphatools.tl.stats import nan_safe_bh_correction
    >>> pvals = np.array([0.01, 0.05, np.nan, 0.001, np.nan])
    >>> corrected = nan_safe_bh_correction(pvals)
    >>> # Returns [0.015, 0.05, nan, 0.015, nan] (approximately)
    """
    # Convert to numpy array if not already
    pvals = np.asarray(pvals)

    # Create output array filled with NaNs
    corrected_pvals = np.full_like(pvals, np.nan, dtype=np.float64)

    # Find indices of non-NaN values
    valid_mask = ~np.isnan(pvals)
    valid_indices = np.where(valid_mask)[0]

    # If there are valid p-values, apply BH correction
    if len(valid_indices) > 0:
        valid_pvals = pvals[valid_mask]
        corrected_valid = false_discovery_control(valid_pvals, method="bh")

        # Put corrected values back in their original positions
        corrected_pvals[valid_indices] = corrected_valid

    return corrected_pvals
