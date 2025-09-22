import anndata
import pandas as pd
import pytest

from alphatools.pl.figure import create_figure
from alphatools.pl.plots import extract_columns_to_df, label_plot


# Test the labelling function of alphatools: correctly spaced and ordered labels
@pytest.fixture
def example_ax():
    def make_dummy_data():
        fig, axm = create_figure(1, 2, figsize=(6, 3))
        ax = axm.next()
        return fig, ax

    return make_dummy_data()


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
        "x",
        "y",
        "labels",
        "anchors",
        "expected_lines",
    ),
    [
        (
            [2, 1, 2, 1, 2, 1],
            [2, 2, 3, 3, 1, 1],
            ["middle_right", "middle_left", "top_right", "top_left", "bottom_right", "bottom_left"],
            None,
            # Expected lines read from plot visually
            pd.DataFrame(
                {
                    "x_start": [1, 2, 1, 2, 1, 2],
                    "x_end": [1, 2, 1, 2, 1, 2],
                    "y_start": [3, 3, 2, 2, 1, 1],
                    "y_end": [3, 3, 2, 2, 1, 1],
                    "label": ["top_left", "top_right", "middle_left", "middle_right", "bottom_left", "bottom_right"],
                }
            ),
        ),
        (
            [2, 1, 2, 1, 2, 1],
            [2, 2, 3, 3, 1, 1],
            ["middle_right", "middle_left", "top_right", "top_left", "bottom_right", "bottom_left"],
            (0.5, 2.5),
            # Expected lines read from plot visually
            pd.DataFrame(
                {
                    "x_start": [1, 1, 1, 2, 2, 2],
                    "x_end": [0.5, 0.5, 0.5, 2.5, 2.5, 2.5],
                    "y_start": [3, 2, 1, 3, 2, 1],
                    "y_end": [3.094, 2.788, 2.484, 3.094, 2.788, 2.484],
                    "label": ["top_left", "middle_left", "bottom_left", "top_right", "middle_right", "bottom_right"],
                }
            ),
        ),
    ],
)
def test_label_plot(example_ax, x, y, labels, anchors, expected_lines):
    _, ax = example_ax

    # Empirical parameters to handle default alphatools font size
    A_DISPLAY_START = 3.20
    Y_PADDING_FACTOR = 10

    # Add the lines to the axes
    label_plot(
        ax=ax,
        x_values=x,
        y_values=y,
        labels=labels,
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

    # Assert that the labels are approximately correct
    pd.testing.assert_frame_equal(label_lines, expected_lines)


# Test the data extraction function of alphatools: correctly extract quantitative data for plotting from dataframes and anndata objects
@pytest.fixture
def example_data():
    def make_dummy_data():
        X = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [7, 8, 9],
            }
        )
        X.index = ["cell1", "cell2", "cell3"]
        return X

    return make_dummy_data()


# example sample metadata: one more sample than data
@pytest.fixture
def example_sample_metadata():
    def make_dummy_data():
        sample_metadata = pd.DataFrame({"cell_type": ["A", "B", "C"], "age": [10, 20, 30]})
        sample_metadata.index = ["cell1", "cell2", "cell3"]
        return sample_metadata

    return make_dummy_data()


@pytest.mark.parametrize(
    ("which_data", "columns", "expected_data"),
    [
        (
            "anndata",
            ["A", "B", "age"],
            pd.DataFrame(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "age": [10, 20, 30],
                },
                index=["cell1", "cell2", "cell3"],
            ),
        ),
        (
            "dataframe",
            ["A", "B"],
            pd.DataFrame(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                },
                index=["cell1", "cell2", "cell3"],
            ),
        ),
    ],
)
def test_extract_columns_to_df(which_data, example_data, example_sample_metadata, columns, expected_data):
    if which_data == "anndata":
        adata = anndata.AnnData(X=example_data, obs=example_sample_metadata)
        data_input = adata
    else:
        data_input = example_data

    extracted_data = extract_columns_to_df(data_input, columns)

    # Ensure the extracted data matches the expected data
    pd.testing.assert_frame_equal(extracted_data, expected_data)


# Test failure cases for extract_columns_to_df
@pytest.mark.parametrize(
    ("which_data", "columns"),
    [
        # Missing column in DataFrame
        ("dataframe", ["A", "nonexistent"]),
        # Missing column in AnnData
        ("anndata", ["A", "nonexistent"]),
        # Duplicate column in both X and obs for AnnData
        ("anndata_with_duplicate", ["A", "age"]),
    ],
)
def test_extract_quan_data_failures(which_data, example_data, example_sample_metadata, columns):
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
        extract_columns_to_df(data_input, columns)
