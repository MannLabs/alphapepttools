import pytest

from alphatools.pl.figure import create_figure
from alphatools.pl.plots import label_plot


# Test the labelling function of alphatools: correctly spaced and ordered labels
@pytest.fixture
def example_ax():
    def make_dummy_data():
        fig, axm = create_figure(1, 2, figsize=(6, 3))
        ax = axm.next()
        return fig, ax

    return make_dummy_data()


# Helper function to read out lines from a matplotlib Axes object
def extract_label_plot_data(ax):
    """Extract line and label data from an axes after label_plot has been called."""
    lines = ax.get_lines()
    texts = ax.texts

    result = []
    for line, text in zip(lines, texts, strict=False):
        x_data = tuple(line.get_xdata())
        y_data = tuple(line.get_ydata())
        label = text.get_text()
        result.append((x_data, y_data, label))
    return result


# The important thing to assess is whether labels and values stay matched throughout the repositioning,
# Hence the providing of labels and values out of order.
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
            [
                ((1, 1), (3, 3), "top_left"),
                ((2, 2), (3, 3), "top_right"),
                ((1, 1), (2, 2), "middle_left"),
                ((2, 2), (2, 2), "middle_right"),
                ((1, 1), (1, 1), "bottom_left"),
                ((2, 2), (1, 1), "bottom_right"),
            ],
        ),
        (
            [2, 1, 2, 1, 2, 1],
            [2, 2, 3, 3, 1, 1],
            ["middle_right", "middle_left", "top_right", "top_left", "bottom_right", "bottom_left"],
            (0.5, 2.5),
            # These values were manually read out from a plot in a test notebook; the important thing
            # here is the order of label y values: top > middle > bottom for left and right
            [
                ((1, 0.5), (3, 3.094), "top_left"),
                ((1, 0.5), (2, 2.788), "middle_left"),
                ((1, 0.5), (1, 2.484), "bottom_left"),
                ((2, 2.5), (3, 3.094), "top_right"),
                ((2, 2.5), (2, 2.788), "middle_right"),
                ((2, 2.5), (1, 2.484), "bottom_right"),
            ],
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

    # Assert that the labels are approximately correct
    for generated_line, expected_line in zip(label_lines, expected_lines, strict=False):
        for i, (gen_i, exp_i) in enumerate(zip(generated_line, expected_line, strict=False)):
            if i == 0:  # x-coordinates
                assert gen_i == exp_i
            elif i == 1:  # y-coodrinates
                assert gen_i == pytest.approx(exp_i, rel=1e-1)
            else:  # labels
                assert gen_i == exp_i
