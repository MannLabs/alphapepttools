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

    if expected_array is None:
        with pytest.raises((ValueError, TypeError)):
            _get_colors_from_cmap(cmap_name=palette, values=input_values)
        return

    result = _get_colors_from_cmap(cmap_name=palette, values=input_values)

    if isinstance(result, list):
        result = np.array(result)

    expected_array = np.array(expected_array)
    np.testing.assert_allclose(result, expected_array, rtol=1e-5), result


# Test for setting custom vmin/vmax value:  _get_colors_from_cmap(..., vmin, vmax) must equal cmap(Normalize(vmin, vmax)(values)).
@pytest.mark.parametrize(
    ("values", "vmin", "vmax", "expected_fractions"),
    [
        # vmin and vmax fully contain the data range: bounds must be vmin/vmax, not data min/max
        (np.array([0.25, 0.5, 0.75]), 0.0, 1.0, [0.25, 0.5, 0.75]),
        # vmin excedes lowest data value but vmax is the same as the data max: data max reaches 1.0
        (np.array([2.0, 4.0]), 0.0, 4.0, [0.5, 1.0]),
        # Data values outside vmin/vmax: clamp to the cmap endpoints: colors don't go beyond the colormap range set by vmin/vmax
        (np.array([-1.0, 0.5, 2.0]), 0.0, 1.0, [0.0, 0.5, 1.0]),
        # Bounds default to data min/max when omitted (unchanged legacy behavior).
        (np.array([2.0, 4.0]), None, None, [0.0, 1.0]),
    ],
)
def test_get_colors_from_cmap_vmin_vmax(values, vmin, vmax, expected_fractions):
    """Explicit vmin/vmax override the data-derived normalization range."""

    cmap = plt.get_cmap("viridis")
    result = _get_colors_from_cmap(cmap_name=cmap, values=values, vmin=vmin, vmax=vmax)
    expected = cmap(np.array(expected_fractions))
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_mappedcolormaps_fit_sets_bounds():
    """fit() stores data-derived bounds and returns self for chaining."""
    mapper = MappedColormaps(cmap="viridis")
    returned = mapper.fit(np.array([2.0, 5.0, 8.0]))

    assert returned is mapper
    assert mapper.vmin == 2.0  # noqa: PLR2004
    assert mapper.vmax == 8.0  # noqa: PLR2004


def test_mappedcolormaps_fit_explicit_bounds_including_zero():
    """Explicit vmin/vmax override data-derived bounds; a zero bound is honored."""
    mapper = MappedColormaps(cmap="viridis")
    # data ranges 2..4, but we pin to [0, 1] — the zero must not be dropped
    mapper.fit(np.array([2.0, 4.0]), vmin=0.0, vmax=1.0)

    assert mapper.vmin == 0.0
    assert mapper.vmax == 1.0


def test_mappedcolormaps_fit_no_data_no_bounds_raises():
    """fit() with neither data nor a full pair of bounds cannot set the range."""
    mapper = MappedColormaps(cmap="viridis")
    with pytest.raises(ValueError):
        mapper.fit()


def test_mappedcolormaps_transform_before_fit_raises():
    """transform() requires bounds from a prior fit()/fit_transform()."""
    mapper = MappedColormaps(cmap="viridis")
    with pytest.raises(ValueError):
        mapper.transform(np.array([0.0, 1.0]))


def test_mappedcolormaps_transform_respects_fitted_bounds():
    """transform() uses the fitted vmin/vmax, not the data range of its argument."""
    cmap = plt.get_cmap("viridis")
    mapper = MappedColormaps(cmap="viridis")
    mapper.fit(vmin=0.0, vmax=1.0)

    # [0.25, 0.75] must land on cmap fractions [0.25, 0.75], not be re-stretched to [0, 1]
    result = mapper.transform(np.array([0.25, 0.75]))
    expected = cmap(np.array([0.25, 0.75]))
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize(
    ("percentile", "as_hex"),
    [(None, False), (None, True), ((5, 95), False), ((5, 95), True)],
)
def test_mappedcolormaps_fit_transform_equals_fit_then_transform(percentile, as_hex):
    """fit_transform(x) must equal fit(x).transform(x) so the paths cannot drift."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=100)

    combined = MappedColormaps(cmap="viridis", percentile=percentile).fit_transform(data, as_hex=as_hex)
    split = MappedColormaps(cmap="viridis", percentile=percentile).fit(data).transform(data, as_hex=as_hex)

    if as_hex:
        assert list(combined) == list(split)
    else:
        np.testing.assert_allclose(combined, split, rtol=1e-5)
