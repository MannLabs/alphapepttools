import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from alphapepttools.pl import defaults
from alphapepttools.pl.colors import (
    BaseColormaps,
    BaseColors,
    MappedColormaps,
    _get_colors_from_cmap,
    get_color_mapping,
)

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

    expected_dict["nan"] = BaseColors.get("lightgrey")

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
    from alphapepttools.pl.colors import _get_colors_from_cmap

    if expected_array is None:
        with pytest.raises((ValueError, TypeError)):
            _get_colors_from_cmap(cmap_name=palette, values=input_values)
        return

    result = _get_colors_from_cmap(cmap_name=palette, values=input_values)

    if isinstance(result, list):
        result = np.array(result)

    expected_array = np.array(expected_array)
    np.testing.assert_allclose(result, expected_array, rtol=1e-5), result


### Test MappedColormaps — verify that data values map correctly to colors ###


class TestMappedColormaps:
    def test_init_sets_cmap_and_resets_bounds(self):
        """Constructor should resolve the colormap and leave vmin/vmax unset until fit_transform."""
        mapper = MappedColormaps(cmap="sequential", percentile=(5, 95))

        assert mapper.cmap is BaseColormaps.get("sequential")
        assert mapper.percentile == (5, 95)
        assert mapper.vmin is None
        assert mapper.vmax is None

    def test_fit_transform_full_range_matches_direct_cmap(self):
        """With percentile=None, the colors returned should be identical to applying the
        colormap directly to the (unclipped) data — i.e. value → color mapping is preserved."""
        data = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        mapper = MappedColormaps(cmap="sequential")

        colors = mapper.fit_transform(data)
        expected = _get_colors_from_cmap(BaseColormaps.get("sequential"), data)

        np.testing.assert_allclose(colors, expected, rtol=1e-6)
        assert mapper.vmin == 0.0
        assert mapper.vmax == 100.0  # noqa: PLR2004

    def test_fit_transform_percentile_clips_outliers_to_same_color(self):
        """Values above the upper percentile (or below the lower) should be clipped — i.e.
        every outlier on the high side should end up with the *same* color as the upper bound.
        This is the core promise of percentile-based normalization."""
        # 50th percentile of this data is 3, so 1000 and 2000 both clip to 3
        data = np.array([1.0, 2.0, 3.0, 1000.0, 2000.0])
        mapper = MappedColormaps(cmap="sequential", percentile=(0, 50))

        colors = mapper.fit_transform(data)

        # vmin/vmax recorded from percentile bounds, not from raw min/max
        assert mapper.vmax == 3.0  # noqa: PLR2004
        # Both outliers (indices -2, -1) clip to vmax=3 → should share the same color
        np.testing.assert_allclose(colors[-1], colors[-2], rtol=1e-6)
        # And that clipped color should match the color of an in-range value at the bound
        np.testing.assert_allclose(colors[-1], colors[2], rtol=1e-6)  # data[2] == 3 == vmax

    def test_fit_transform_identical_values_get_identical_colors(self):
        """Duplicate values should map to identical colors."""
        data = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
        mapper = MappedColormaps(cmap="sequential")

        colors = mapper.fit_transform(data)

        # data[0] == data[2] → same color
        np.testing.assert_allclose(colors[0], colors[2], rtol=1e-6)
        # data[1] == data[4] → same color
        np.testing.assert_allclose(colors[1], colors[4], rtol=1e-6)
        # different values produce different colors (sanity)
        assert not np.allclose(colors[0], colors[1])

    def test_fit_transform_as_hex_returns_hex_strings(self):
        """`as_hex=True` should produce hex color strings of the right shape."""
        data = np.array([1.0, 5.0, 10.0])
        mapper = MappedColormaps(cmap="sequential")

        hex_colors = mapper.fit_transform(data, as_hex=True)

        assert hex_colors.shape == (3,)
        assert all(isinstance(c, str) and c.startswith("#") for c in hex_colors)

    def test_scalar_mappable_raises_before_fit_transform(self):
        """Accessing scalar_mappable before fit_transform should raise."""
        mapper = MappedColormaps(cmap="sequential")
        with pytest.raises(ValueError, match="fit_transform must be called"):
            _ = mapper.scalar_mappable

    def test_scalar_mappable_after_fit_transform(self):
        """After fit_transform, scalar_mappable should expose the same vmin/vmax and cmap."""
        data = np.array([2.0, 4.0, 6.0, 8.0])
        mapper = MappedColormaps(cmap="sequential")
        mapper.fit_transform(data)

        sm = mapper.scalar_mappable

        assert isinstance(sm, mpl.cm.ScalarMappable)
        assert sm.norm.vmin == mapper.vmin
        assert sm.norm.vmax == mapper.vmax
        assert sm.cmap is mapper.cmap
