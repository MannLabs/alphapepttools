import matplotlib.pyplot as plt
import numpy as np
import pytest

from alphatools.pl import defaults
from alphatools.pl.colors import BaseColormaps, BaseColors, get_color_mapping

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


# Test _get_colors_from_cmap, which maps values quantitatively
@pytest.mark.parametrize(
    ("input_values", "palette", "expected_array"),
    [
        # Numeric values with NaN, disordered, to palette
        (
            np.array([3.7, 1.5, 23.5, np.nan, 1.5]),
            BaseColormaps.get("sequential"),
            np.array(
                [
                    [0.162198, 0.191754, 0.391238, 1.0],
                    [0.171032, 0.100402, 0.299782, 1.0],
                    [0.999916, 0.99997, 0.999952, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.171032, 0.100402, 0.299782, 1.0],
                ]
            ),
        ),
        # int of requested colors
        (
            len(np.array([3.7, 1.5, 23.5, np.nan, 1.5])),
            BaseColormaps.get("sequential"),
            [
                (0.171032, 0.100402, 0.299782, 1.0),
                (0.159616, 0.344655, 0.560957, 1.0),
                (0.495894, 0.558869, 0.866797, 1.0),
                (0.796559, 0.778108, 0.958163, 1.0),
                (0.999916, 0.99997, 0.999952, 1.0),
            ],
        ),
        # fail if input is non-numeric
        (
            np.array(["cherry", "apple", "banana", "apple", np.nan]),
            BaseColormaps.get("sequential"),
            None,
        ),
    ],
)
def test_get_colors_from_cmap(input_values, palette, expected_array):
    """Test _get_colors_from_cmap with various input types and edge cases."""
    from alphatools.pl.colors import _get_colors_from_cmap

    if expected_array is None:
        with pytest.raises((ValueError, TypeError)):
            _get_colors_from_cmap(cmap_name=palette, values=input_values)
        return

    result = _get_colors_from_cmap(cmap_name=palette, values=input_values)

    if isinstance(result, list):
        result = np.array(result)

    expected_array = np.array(expected_array)
    np.testing.assert_allclose(result, expected_array, rtol=1e-5), result
