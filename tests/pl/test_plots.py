import pytest

from alphatools.pl.figure import create_figure
from alphatools.pl.plots import label_plot


# Test the labelling function of alphatools: correctly spaced and ordered labels
@pytest.fixture
def example_ax():
    def make_dummy_data():
        fig, axm = create_figure(1, 2, figsize=(6, 3))
        return fig, axm.next()

    return make_dummy_data()


# 3 --> 3.057
# 2 --> 2.711
# 1 --> 2.365
# The important thing to assess is whether labels and values stay matched throughout the repositioning
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
            [
                ((1, 0.5), (3, 3.057), "top_left"),
                ((1, 0.5), (2, 2.711), "middle_left"),
                ((1, 0.5), (1, 2.365), "bottom_left"),
                ((2, 2.5), (3, 3.057), "top_right"),
                ((2, 2.5), (2, 2.711), "middle_right"),
                ((2, 2.5), (1, 2.365), "bottom_right"),
            ],
        ),
    ],
)
def test_label_plot(example_ax, x, y, labels, anchors, expected_lines):
    fig, ax = example_ax

    # Empirical parameters to handle default alphatools font size
    A_DISPLAY_START = 3.20
    Y_PADDING_FACTOR = 10

    # Generate the lines
    label_lines = label_plot(
        ax=ax,
        x_values=x,
        y_values=y,
        labels=labels,
        x_anchors=anchors,
        y_display_start=A_DISPLAY_START,
        y_padding_factor=Y_PADDING_FACTOR,
        line_operation="add_return",
    )

    # Assert that the labels are approximately correct
    for generated_line, expected_line in zip(label_lines, expected_lines, strict=False):
        for i, (gen_i, exp_i) in enumerate(zip(generated_line, expected_line, strict=False)):
            if i == 0:  # x-coordinates
                assert gen_i == exp_i
            elif i == 1:  # y-coodrinates
                assert gen_i == pytest.approx(exp_i, rel=1e-1)  # Fails on second, unexplained?
            else:  # labels
                assert gen_i == exp_i
