import matplotlib.pyplot as plt
import numpy as np
import pytest

from alphatools.pl import defaults
from alphatools.pl.colors import BaseColors, get_color_mapping

config = defaults.plot_settings.to_dict()


@pytest.mark.parametrize(
    ("input_values", "palette", "expected_dict"),
    [
        # Numeric values with NaN, disordered, to palette
        (
            np.array([3.7, 1.5, 2.3, np.nan, 1.5]),
            ["red", "green"],
            {"1.5": "red", "2.3": "green", "3.7": "red"},
        ),
        # Object with NaN, disordered, to palette
        (
            np.array(["cherry", "apple", "banana", "apple", np.nan]),
            ["red", "green"],
            {"apple": "red", "banana": "green", "cherry": "red"},
        ),
        # Numeric values with NaN, disordered, to colormap
        (
            np.array([3.7, 1.5, 2.3, np.nan, 1.5]),
            plt.get_cmap("viridis"),
            {
                "1.5": (0.267004, 0.004874, 0.329415, 1.0),
                "2.3": (0.127568, 0.566949, 0.550556, 1.0),
                "3.7": (0.993248, 0.906157, 0.143936, 1.0),
            },
        ),
        # Object with NaN, disordered, to colormap
        (
            np.array(["cherry", "apple", "banana", "apple", np.nan]),
            plt.get_cmap("viridis"),
            {
                "apple": (0.267004, 0.004874, 0.329415, 1.0),
                "banana": (0.127568, 0.566949, 0.550556, 1.0),
                "cherry": (0.993248, 0.906157, 0.143936, 1.0),
            },
        ),
    ],
)
def test_get_color_mapping(input_values, palette, expected_dict):
    """Test get_color_mapping with various input types and edge cases."""
    expected_dict = dict(expected_dict.items())

    expected_dict[config["na_default"]] = BaseColors.get("lightgrey")

    result = get_color_mapping(input_values, palette)

    assert result == expected_dict
