# Tools for data processing

import logging

import numpy as np
from scipy.stats import gaussian_kde

# logging configuration
logging.basicConfig(level=logging.INFO)


def umap() -> None:
    """Perform UMAP on the data"""
    raise NotImplementedError


def gaussian_density(
    x: np.array,
    y: np.array,
) -> np.ndarray:
    """Estimate the Gaussian density of the data points

    Parameters
    ----------
    x : np.ndarray
        x-coordinates of the data points
    y : np.ndarray
        y-coordinates of the data points

    Returns
    -------
    np.ndarray
        Gaussian density values for the data points

    """
    # create correct shape for gaussian density
    xz = np.vstack([x, y])

    # estimate density
    pdf = gaussian_kde(xz)

    # get density values
    return pdf(xz)
