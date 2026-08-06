import logging

import anndata
import numpy as np
import pandas as pd
import pytest

from alphapepttools.pl.figure import create_figure
from alphapepttools.pl.plots import (
    PlotConfig,
    _array_to_str,
    _assign_nearest_anchor_position_to_values,
    _dict_keys_to_str,
    _extract_groupwise_plotting_data,
    _extract_plot_layer_specs,
    drop_nan_coordinate_points,
    label_plot,
    make_scatter_config,
)
from alphapepttools.pp.data import data_columns_to_df


# Fixtures
@pytest.fixture
def example_data():
    def make_dummy_data():
        X = pd.DataFrame(
            {
                "A": [np.nan, 2.0, 3.0],
                "B": [4.0, 5.0, 6.0],
                "C": [7.0, 8.0, 9.0],
            }
        )
        X.index = ["cell1", "cell2", "cell3"]
        return X

    return make_dummy_data()


@pytest.fixture
def example_sample_metadata():
    def make_dummy_data():
        sample_metadata = pd.DataFrame({"cell_type": ["A", "B", "C"], "age": [10.0, 20.0, 30.0], "batch": [1, 1, 2]})
        sample_metadata.index = ["cell1", "cell2", "cell3"]
        return sample_metadata

    return make_dummy_data()


@pytest.fixture
def example_ax():
    def make_dummy_data():
        fig, axm = create_figure(1, 2, figsize=(6, 3))
        ax = axm.next()
        return fig, ax

    return make_dummy_data()


# Test the labelling function of alphapepttools: correctly spaced and ordered labels
def extract_label_plot_data(ax):
    """Extract line and label data from an axes after label_plot has been called."""
    lines = ax.get_lines()
    texts = ax.texts

    line_dfs = []
    for line, text in zip(lines, texts, strict=False):
        x_left, x_right = line.get_xdata()
        y_left, y_right = line.get_ydata()
        label = text.get_text()
        line_dfs.append(
            pd.DataFrame(
                {
                    "x_start": [x_left],
                    "x_end": [x_right],
                    "y_start": [y_left],
                    "y_end": [y_right],
                    "label": [label],
                }
            )
        )

    return pd.concat(line_dfs, ignore_index=True)


# The important thing to assess is that x, y and labels stay correctly ordered, i.e. top_right
# ends up at the top right label after anchor assignment.
@pytest.mark.parametrize(
    (
        "data",
        "x_column",
        "y_column",
        "label_column",
        "anchors",
        "expected_lines",
    ),
    [
        (
            pd.DataFrame(
                {
                    "x": [2, 1, 2, 1, 2, 1],
                    "y": [2, 2.1, 3, 3.1, 1, 1.1],
                    "label": ["middle_right", "middle_left", "top_right", "top_left", "bottom_right", "bottom_left"],
                }
            ),
            "x",
            "y",
            "label",
            None,
            # Expected lines read from plot visually
            pd.DataFrame(
                {
                    "x_start": [1, 2, 1, 2, 1, 2],
                    "x_end": [1, 2, 1, 2, 1, 2],
                    "y_start": [3.1, 3.0, 2.1, 2.0, 1.1, 1.0],
                    "y_end": [3.1, 3.0, 2.1, 2.0, 1.1, 1.0],
                    "label": ["top_left", "top_right", "middle_left", "middle_right", "bottom_left", "bottom_right"],
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "x": [2, 1, 2, 1, 2, 1],
                    "y": [2, 2.1, 3, 3.1, 1, 1.1],
                    "label": ["middle_right", "middle_left", "top_right", "top_left", "bottom_right", "bottom_left"],
                }
            ),
            "x",
            "y",
            "label",
            (0.5, 2.5),
            # Expected lines read from plot visually
            pd.DataFrame(
                {
                    "x_start": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                    "x_end": [0.5, 0.5, 0.5, 2.5, 2.5, 2.5],
                    "y_start": [3.1, 2.1, 1.1, 3.0, 2.0, 1.0],
                    "y_end": [
                        3.0571428571428574,
                        2.686502261783161,
                        2.315861666423465,
                        3.0571428571428574,
                        2.686502261783161,
                        2.315861666423465,
                    ],
                    "label": ["top_left", "middle_left", "bottom_left", "top_right", "middle_right", "bottom_right"],
                }
            ),
        ),
    ],
)
def test_label_plot(example_ax, data, x_column, y_column, label_column, anchors, expected_lines):
    _, ax = example_ax

    # Empirical parameters to handle default alphapepttools font size
    A_DISPLAY_START = 3.20
    Y_PADDING_FACTOR = 10

    # Add the lines to the axes
    label_plot(
        ax=ax,
        data=data,
        x_column=x_column,
        y_column=y_column,
        label_column=label_column,
        x_anchors=anchors,
        y_display_start=A_DISPLAY_START,
        y_padding_factor=Y_PADDING_FACTOR,
    )

    # Extract the actual lines
    label_lines = extract_label_plot_data(ax)

    # For both dataframes, for each x anchor (x_end) convert the y_end points to ranks to avoid issues with absolute positioning
    label_lines["y_end"] = label_lines.groupby("x_end")["y_end"].rank(ascending=True)
    expected_lines["y_end"] = expected_lines.groupby("x_end")["y_end"].rank(ascending=True)

    # Set datatypes
    comparison_datatypes = {
        "x_start": float,
        "x_end": float,
        "y_start": float,
        "y_end": float,
        "label": str,
    }

    label_lines = label_lines.astype(comparison_datatypes)
    expected_lines = expected_lines.astype(comparison_datatypes)

    # Assert that the labels are approximately correct (ignoring row order)
    pd.testing.assert_frame_equal(label_lines, expected_lines, check_like=True)


# Test data extraction for plotting from dataframes and anndata objects
@pytest.mark.parametrize(
    ("which_data", "columns", "expected_data"),
    [
        (
            "anndata",
            ["A", "B", "age"],
            pd.DataFrame(
                {
                    "A": [np.nan, 2.0, 3.0],
                    "B": [4.0, 5.0, 6.0],
                    "age": [10.0, 20.0, 30.0],
                },
                index=["cell1", "cell2", "cell3"],
            ),
        ),
        (
            "dataframe",
            ["A", "B"],
            pd.DataFrame(
                {
                    "A": [np.nan, 2.0, 3.0],
                    "B": [4.0, 5.0, 6.0],
                },
                index=["cell1", "cell2", "cell3"],
            ),
        ),
    ],
)
def test_data_columns_to_df(which_data, example_data, example_sample_metadata, columns, expected_data):
    if which_data == "anndata":
        adata = anndata.AnnData(X=example_data, obs=example_sample_metadata)
        data_input = adata
    else:
        data_input = example_data

    extracted_data = data_columns_to_df(data_input, columns)

    pd.testing.assert_frame_equal(extracted_data, expected_data)


# Test failure cases for extract_columns_to_df
@pytest.mark.parametrize(
    ("which_data", "columns"),
    [
        ("dataframe", ["A", "nonexistent"]),
        ("anndata", ["A", "nonexistent"]),
        ("anndata_with_duplicate", ["A", "age"]),
    ],
)
def test_data_columns_to_df_failures(which_data, example_data, example_sample_metadata, columns):
    if which_data == "anndata":
        adata = anndata.AnnData(X=example_data, obs=example_sample_metadata)
        data_input = adata
    elif which_data == "anndata_with_duplicate":
        X_with_age = example_data.copy()
        X_with_age.columns = ["A", "B", "age"]
        adata = anndata.AnnData(X=X_with_age, obs=example_sample_metadata)
        data_input = adata
    else:
        data_input = example_data

    with pytest.raises(KeyError):
        data_columns_to_df(data_input, columns)


# Test parsing of anndata objects to bar/box/violin-plottable data
@pytest.mark.parametrize(
    ("grouping_column", "value_column", "direct_columns", "expected_data", "expected_labels", "expected_positions"),
    [
        # Basic functionality with grouping and value column
        (
            "batch",
            "A",
            None,
            [[2.0], [3.0]],  # NaN is dropped, grouped by batch [1,1,2]
            [1, 2],
            [0, 1],
        ),
        (
            "batch",
            "B",
            None,
            [[4.0, 5.0], [6.0]],  # First two cells in batch 1, last in batch 2
            [1, 2],
            [0, 1],
        ),
        # Case with direct column usage and dropping NaNs
        (
            None,
            None,
            ["A", "B", "age"],
            [[2.0, 3.0], [4.0, 5.0, 6.0], [10.0, 20.0, 30.0]],
            ["A", "B", "age"],
            [0, 1, 2],
        ),
    ],
)
def test__extract_groupwise_plotting_data(
    example_data,
    example_sample_metadata,
    grouping_column,
    value_column,
    direct_columns,
    expected_data,
    expected_labels,
    expected_positions,
):
    adata = anndata.AnnData(X=example_data, obs=example_sample_metadata)

    data_lists, labels, positions, color_keys = _extract_groupwise_plotting_data(
        data=adata, grouping_column=grouping_column, value_column=value_column, direct_columns=direct_columns
    )

    assert data_lists == expected_data
    assert labels == expected_labels
    assert positions == expected_positions
    # In flat mode color_keys mirrors labels so a label-keyed color_dict still works.
    assert color_keys == expected_labels


### Test drop_nan_coordinate_points — alignment contract ###
def test_drop_nan_coordinate_points_keeps_arrays_aligned():
    """NaN in x or y at index i should drop labels[i] in lockstep — same length, same order."""
    x = np.array([1.0, np.nan, 3.0, 4.0])
    y = np.array([5.0, 6.0, np.nan, 8.0])
    labels = np.array(["a", "b", "c", "d"])

    x_out, y_out, labels_out = drop_nan_coordinate_points(x, y, labels)

    # Indices 1 (x is NaN) and 2 (y is NaN) should be dropped; 0 and 3 kept
    assert np.array_equal(x_out, np.array([1.0, 4.0]))
    assert np.array_equal(y_out, np.array([5.0, 8.0]))
    assert np.array_equal(labels_out, np.array(["a", "d"]))
    # Pin the alignment contract: all three arrays end up the same length
    assert len(x_out) == len(y_out) == len(labels_out)


def test_drop_nan_coordinate_points_preserves_order_with_scattered_nans():
    """NaNs scattered across both x and y should not reorder the surviving rows."""
    x = np.array([np.nan, 2.0, 3.0, np.nan, 5.0])
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    labels = np.array(["first", "second", "third", "fourth", "fifth"])

    x_out, y_out, labels_out = drop_nan_coordinate_points(x, y, labels)

    # Only indices 1 and 4 survive; original order preserved
    assert np.array_equal(x_out, np.array([2.0, 5.0]))
    assert np.array_equal(y_out, np.array([2.0, 5.0]))
    assert np.array_equal(labels_out, np.array(["second", "fifth"]))


### Test _assign_nearest_anchor_position_to_values ###


def test_assign_nearest_anchor_position_to_values_none_anchors():
    """When anchors is None, values should be returned unchanged."""
    values = np.array([1.2, 2.7, 5.1])

    result = _assign_nearest_anchor_position_to_values(values, anchors=None)

    assert result is values  # no-op short-circuit


def test_assign_nearest_anchor_position_to_values_snaps_and_preserves_order():
    """Each input value should snap to its nearest anchor; output order matches input order."""
    values = np.array([1.2, 5.1, 2.7])  # deliberately out of sorted order
    anchors = [1, 3, 5]

    result = _assign_nearest_anchor_position_to_values(values, anchors)

    # 1.2 → 1, 5.1 → 5, 2.7 → 3 (input order preserved)
    assert np.array_equal(result, np.array([1, 5, 3]))


### Test _extract_groupwise_plotting_data conflicting-args warning ###


def test_extract_groupwise_plotting_data_warns_on_conflicting_args(example_data, example_sample_metadata, caplog):
    """If both `direct_columns` and `grouping_column`/`value_column` are passed, log info
    and use `direct_columns`."""
    adata = anndata.AnnData(X=example_data, obs=example_sample_metadata)

    with caplog.at_level(logging.INFO):
        _extract_groupwise_plotting_data(
            data=adata,
            grouping_column="batch",
            value_column="A",
            direct_columns=["A", "B"],
        )

    assert "ignoring 'grouping_column' and 'value_column'" in caplog.text


### Test _array_to_str ###


def test_array_to_str_coerces_to_string():
    """Non-string array elements should be coerced to strings."""
    result = _array_to_str(np.array([1, 2, 3]))
    assert result.dtype.kind == "U"  # numpy unicode string
    assert result.tolist() == ["1", "2", "3"]


def test_array_to_str_handles_series():
    """pd.Series input is also accepted."""
    result = _array_to_str(pd.Series([1.5, 2.5]))
    assert result.tolist() == ["1.5", "2.5"]


### Test _dict_keys_to_str ###


def test_dict_keys_to_str_coerces_keys():
    """Non-string dict keys should be coerced to strings; values are passed through unchanged."""
    result = _dict_keys_to_str({1: "red", 2: "blue"})
    assert result == {"1": "red", "2": "blue"}


def test_dict_keys_to_str_already_strings():
    """String keys should remain unchanged."""
    result = _dict_keys_to_str({"x": 1, "y": 2})
    assert result == {"x": 1, "y": 2}


### Test PlotConfig ###


class TestPlotConfig:
    def test_post_init_none_extra_becomes_empty_dict(self):
        """When `_extra` is not passed, it should be initialized as an empty dict."""
        config = PlotConfig(data=None)
        assert config._extra == {}

    def test_getattr_returns_extra_field(self):
        """Fields stored in `_extra` should be accessible as attributes."""
        config = PlotConfig(data=None, _extra={"color": "red", "size": 10})
        assert config.color == "red"
        assert config.size == 10  # noqa: PLR2004

    def test_getattr_unknown_field_raises(self):
        """Unknown attribute access should raise KeyError."""
        config = PlotConfig(data=None, _extra={"color": "red"})
        with pytest.raises(KeyError, match="has no attribute"):
            _ = config.nonexistent

    def test_copy_with_overrides_existing_fields(self):
        """`copy_with` should produce a new PlotConfig with the overrides applied."""
        config = PlotConfig(data=None, _extra={"color": "red", "size": 10})
        new_config = config.copy_with(color="blue", marker="o")

        # New values applied, untouched values preserved
        assert new_config.color == "blue"
        assert new_config.size == 10  # noqa: PLR2004
        assert new_config.marker == "o"
        # Original is unchanged (frozen dataclass)
        assert config.color == "red"

    def test_copy_with_overrides_data(self):
        """`copy_with` can override the `data` field."""
        df = pd.DataFrame({"x": [1, 2]})
        config = PlotConfig(data=None, _extra={"a": 1})
        new_config = config.copy_with(data=df)
        pd.testing.assert_frame_equal(new_config.data, df)

    def test_to_kwargs_includes_data_when_set(self):
        """`to_kwargs` should include `data` and unpack `_extra`, dropping None values."""
        df = pd.DataFrame({"x": [1]})
        config = PlotConfig(data=df, _extra={"color": "red", "size": None})
        kwargs = config.to_kwargs()

        pd.testing.assert_frame_equal(kwargs["data"], df)
        assert kwargs["color"] == "red"
        assert "size" not in kwargs  # None values dropped

    def test_to_kwargs_skips_none_data(self):
        """If `data` is None, it should not appear in the kwargs."""
        config = PlotConfig(data=None, _extra={"color": "red"})
        kwargs = config.to_kwargs()
        assert "data" not in kwargs
        assert kwargs == {"color": "red"}


### Test make_scatter_config ###


def test_make_scatter_config_packs_arguments():
    """`make_scatter_config` should return a PlotConfig with the supplied data and extras."""
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    config = make_scatter_config(df, x_column="x", y_column="y", color="red", alpha=0.5)

    assert isinstance(config, PlotConfig)
    pd.testing.assert_frame_equal(config.data, df)
    assert config.x_column == "x"
    assert config.y_column == "y"
    assert config.color == "red"
    assert config.alpha == 0.5  # noqa: PLR2004


### Test _extract_plot_layer_specs ###


def test_extract_plot_layer_specs_three_elements():
    """A 3-element spec returns the three fields plus an empty kwargs dict."""
    result = _extract_plot_layer_specs(("batch", 1, "blue"))
    assert result == ("batch", 1, "blue", {})


def test_extract_plot_layer_specs_four_elements():
    """A 4-element spec returns the kwargs dict from the fourth element."""
    kwargs = {"marker": "x", "s": 30}
    result = _extract_plot_layer_specs(("batch", [1, 2], "blue", kwargs))
    assert result == ("batch", [1, 2], "blue", kwargs)


@pytest.mark.parametrize(
    ("layer_specs", "expected_exception", "match"),
    [
        # Too few elements
        (("batch", 1), ValueError, "at least 3 elements"),
        # Too many elements
        (("batch", 1, "blue", {}, "extra"), ValueError, "at most 4 elements"),
        # layer_column not a string
        ((123, 1, "blue"), TypeError, "layer_column must be str"),
        # layer_val wrong type
        (("batch", 1.5, "blue"), TypeError, "layer_val must be str.int.list"),
        # color_key not a string
        (("batch", 1, 99), TypeError, "color_key must be str"),
        # scatter_kwargs not a dict
        (("batch", 1, "blue", "not_a_dict"), TypeError, "scatter_kwargs must be dict"),
    ],
)
def test_extract_plot_layer_specs_validation(layer_specs, expected_exception, match):
    with pytest.raises(expected_exception, match=match):
        _extract_plot_layer_specs(layer_specs)


def test__extract_groupwise_plotting_data_subgrouped():
    """Subgroup mode dodges positions, emits one cell per (group, subgroup),
    keeps labels per-cell as the main group, and emits subgroup values as color_keys.
    """
    df = pd.DataFrame(
        {
            "precursor": ["P1", "P1", "P1", "P1", "P2", "P2", "P2", "P2"],
            "condition": ["ctrl", "ctrl", "treat", "treat", "ctrl", "ctrl", "treat", "treat"],
            "intensity": [1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0],
        }
    )

    data_lists, labels, positions, color_keys = _extract_groupwise_plotting_data(
        data=df,
        grouping_column="precursor",
        value_column="intensity",
        subgroup_column="condition",
        width=0.4,
    )

    assert data_lists == [[1.0, 2.0], [10.0, 20.0], [3.0, 4.0], [30.0, 40.0]]
    assert labels == ["P1", "P1", "P2", "P2"]
    assert color_keys == ["ctrl", "treat", "ctrl", "treat"]
    # With width=0.4 and 2 subgroups: offsets = [-0.2, +0.2] around each main-group integer.
    assert positions == pytest.approx([-0.2, 0.2, 0.8, 1.2])
