"""Unit tests for the plumbing helpers in alphapepttools.pl.figure.

The visual functions (`stylize`, `label_axes`, `create_figure`, `save_figure`)
are intentionally left untested — bugs there are caught visually. The helpers
covered here normalize shapes and resolve presets, where wrong outputs end up
silently placing content on the wrong axis or producing the wrong figsize.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from alphapepttools.pl.figure import (
    _indexable_axes,
    _parse_figsize,
    create_figure,
)

### Test _indexable_axes — normalize matplotlib's varying return shapes into a 2D array ###


class TestIndexableAxes:
    def test_single_axes_becomes_2d(self):
        """A bare matplotlib Axes should be wrapped into a (1, 1) 2D array."""
        _, ax = plt.subplots(1, 1)

        result = _indexable_axes(ax)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)
        assert result[0, 0] is ax

    def test_1d_array_becomes_2d(self):
        """A 1D numpy array of Axes should be expanded to 2D."""
        _, axs = plt.subplots(1, 3)  # returns 1D array

        result = _indexable_axes(axs)

        assert result.shape == (1, 3)
        # Identity preserved
        for i in range(3):
            assert result[0, i] is axs[i]

    def test_2d_array_passes_through(self):
        """A 2D numpy array should be returned with the same shape."""
        _, axs = plt.subplots(2, 3)  # returns 2D array

        result = _indexable_axes(axs)

        assert result.shape == (2, 3)

    def test_list_becomes_array(self):
        """A Python list of Axes should be converted to a numpy array (and then 2D)."""
        _, axs = plt.subplots(1, 2)
        ax_list = [axs[0], axs[1]]

        result = _indexable_axes(ax_list)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2)

    def test_invalid_type_raises(self):
        """Anything that isn't an Axes/list/ndarray should raise TypeError."""
        with pytest.raises(TypeError, match="Invalid axes"):
            _indexable_axes("not an axes")


### Test _parse_figsize — figsize parsing with preset string keys ###


class TestParseFigsize:
    def test_none_returns_default(self):
        """`None` should resolve to the preset '1' size in inches (both width and height)."""
        result = _parse_figsize(None)

        # preset "1" is 89 mm; converted to inches
        expected = 89 / 25.4
        assert result == (expected, expected)

    def test_string_preset_keys_convert_to_inches(self):
        """A pair of valid preset string keys should resolve to mm/25.4 inches."""
        result = _parse_figsize(("1", "0.5"))

        # preset "1" = 89 mm, "0.5" = 45 mm
        assert result == (89 / 25.4, 45 / 25.4)

    def test_invalid_string_preset_raises(self):
        """An unknown preset string key should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid strings"):
            _parse_figsize(("nope_a", "nope_b"))

    def test_numeric_tuple_passes_through(self):
        """A numeric tuple should be returned as-is (already in inches)."""
        result = _parse_figsize((6.0, 4.0))
        assert result == (6.0, 4.0)

    def test_mixed_types_raises(self):
        """Mixing string and numeric in a single figsize tuple should raise."""
        with pytest.raises(ValueError, match="Invalid figsize"):
            _parse_figsize(("1", 4.0))


### Test AxisManager.__getitem__ — tuple/integer indexing into a styled subplot grid ###


class TestAxisManagerIndexing:
    @pytest.fixture
    def axm_2x3(self):
        """A 2x3 grid of axes to exercise both integer and tuple indexing."""
        _, axm = create_figure(2, 3)
        return axm

    def test_integer_key_returns_axis_in_row_major_order(self, axm_2x3):
        """`axm[0]` is top-left; `axm[3]` is second row, first column."""
        flat = axm_2x3._axs_flat

        assert axm_2x3[0] is flat[0]
        assert axm_2x3[3] is flat[3]

    def test_integer_key_updates_current_index(self, axm_2x3):
        """Integer indexing should advance `current_i` to one past the accessed index."""
        _ = axm_2x3[2]
        assert axm_2x3.current_i == 3  # noqa: PLR2004

    def test_integer_key_out_of_bounds_raises(self, axm_2x3):
        """Out-of-range integer index should raise IndexError."""
        with pytest.raises(IndexError, match="out of bounds"):
            _ = axm_2x3[99]

    def test_tuple_key_returns_correct_subplot(self, axm_2x3):
        """`axm[i, j]` should return the axis at row i, column j of the underlying grid."""
        assert axm_2x3[0, 0] is axm_2x3.axs[0, 0]
        assert axm_2x3[1, 2] is axm_2x3.axs[1, 2]

    def test_tuple_key_updates_current_index_row_major(self, axm_2x3):
        """Tuple indexing should set `current_i` to the row-major offset."""
        # 2 rows x 3 cols; (1, 2) is the bottom-right -> row-major offset 5
        _ = axm_2x3[1, 2]
        assert axm_2x3.current_i == 5  # noqa: PLR2004

    def test_tuple_key_out_of_bounds_raises(self, axm_2x3):
        """An out-of-range tuple index should raise IndexError."""
        with pytest.raises(IndexError, match="out of bounds"):
            _ = axm_2x3[5, 0]


### Test AxisManager.next() exhaustion and reset() ###


def test_axis_manager_reset_returns_to_start():
    """`reset()` should set `current_i` back to 0 so iteration can restart."""
    _, axm = create_figure(1, 3)
    axm.next()
    axm.next()
    assert axm.current_i == 2  # noqa: PLR2004

    axm.reset()

    assert axm.current_i == 0


def test_axis_manager_next_raises_stop_iteration_when_exhausted():
    """Calling `next()` past the end of the grid should raise StopIteration."""
    _, axm = create_figure(1, 2)
    axm.next()
    axm.next()
    with pytest.raises(StopIteration, match="No more axes"):
        axm.next()
