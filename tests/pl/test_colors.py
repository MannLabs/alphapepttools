import numpy as np
import pytest

from alphatools.pl import defaults

config = defaults.plot_settings.to_dict()


@pytest.mark.parametrize(
    ("input_values", "palette", "expected_dict"),
    [
        # Numeric values with NaN, disordered, to palette
        (
            np.array([3.7, 1.5, 2.3, np.nan, 1.5]),
            ["red", "green", "blue"],
            {"1.5": "red", "2.3": "green", "3.7": "blue", "_NA_": "_lightgrey_"},
        ),
        # Object with NaN, disordered, to palette
        (
            np.array(["cherry", "apple", "banana", "apple", np.nan]),
            ["red", "green", "blue"],
            {"apple": "red", "banana": "green", "cherry": "blue", "_NA_": "_lightgrey_"},
            False,
        ),
        #     # Numeric values with NaN, disordered, to colormap
        #     (
        #         np.array([3.7, 1.5, 2.3, np.nan, 1.5]),
        #         plt.get_cmap("viridis"),
        #         {"1.5": "red", "2.3": "green", "3.7": "blue", "_NA_": "_lightgrey_"},
        #     ),
        #     # Object with NaN, disordered, to colormap
        #     (
        #         np.array(["cherry", "apple", "banana", "apple", np.nan]),
        #         ["red", "green", "blue"],
        #         {"apple": "red", "banana": "green", "cherry": "blue", "_NA_": "_lightgrey_"},
        #         False,
        #     ),
    ],
)
def test_get_color_mapping(input_values, palette_type, expected_keys, has_na_color):
    """Test get_color_mapping with various input types and edge cases."""
