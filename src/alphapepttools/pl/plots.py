# plots.py

# Main plotting submodule with module-level plotting functions plus helpers for
# legends and labels. The layout for plotting functions is such that they accept
# AnnData objects and dataframes.
# When columns to plot are specified for an AnnData object, the _adata_column_to_array()
# function first tries to find the column in the var_names (i.e. the columns of the actual
# data), and then in the obs.columns (for example, when plotting a numeric value from X and
# coloring it by a metadata column from obs, see 03_basic_workflow.ipynb).

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import anndata as ad
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from alphapepttools.pl import defaults
from alphapepttools.pl.colors import BaseColors, BasePalettes, _get_colors_from_cmap, get_color_mapping
from alphapepttools.pl.figure import create_figure, label_axes
from alphapepttools.pp.data import (
    _tolist,
    coerce_to_dataframe,
    data_column_to_array,
    data_columns_to_df,
    data_index_to_array,
    subset_data,
)
from alphapepttools.tl.plot_data_handling import (
    extract_pca_anndata,
    prepare_pca_1d_loadings_data_to_plot,
    prepare_pca_2d_loadings_data_to_plot,
    prepare_scree_data_to_plot,
)
from alphapepttools.tl.utils import find_iterable_kwargs

# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = defaults.plot_settings.to_dict()


def _extract_groupwise_plotting_data(
    data: ad.AnnData | pd.DataFrame,
    grouping_column: str | None = None,
    value_column: str | None = None,
    direct_columns: list[str] | None = None,
    subgroup_column: str | None = None,
    width: float = 0.5,
) -> tuple[list[list], list[str], list[float], list[str]]:
    """Extract data for group-wise plotting (violin, bar, box plots)

    Transforms long-format data into the list-of-lists format required by
    matplotlib's violin, bar, and box plot functions. Each sublist contains
    the values for one group.

    When ``subgroup_column`` is provided, cells are emitted per ``(group, subgroup)``
    pair and positions are dodged so that subgroups within a main group sit close
    together while main groups stay spaced one unit apart.

    Parameters
    ----------
    data
        Data containing grouping and value columns
    grouping_column
        Column containing the groups to compare
    value_column
        Column whose values should be plotted
    direct_columns
        Alternative to grouping/value columns: treat each column as a separate group
    subgroup_column
        Optional column whose values subdivide each main group. When provided,
        each ``(group, subgroup)`` pair becomes its own cell with a dodged position.
    width
        Visual width of each cell. In subgrouped mode it is also the step between
        adjacent subgroup positions, so neighbouring subgroup bars sit flush.
        Reduce this value when many subgroups would otherwise overflow into the
        next main group's slot.

    Returns
    -------
    tuple[list[list], list[str], list[float], list[str]]
        (data_lists, labels, positions, color_keys) for plotting. ``color_keys``
        contains group labels when ``subgroup_column`` is None, and subgroup
        labels when ``subgroup_column`` is provided.

    Examples
    --------
    Group by categorical column:

    .. code-block:: python

        df = pd.DataFrame({"treatment": ["A", "A", "B", "B", "C"], "intensity": [1, 2, 3, 4, 5]})

        data_lists, labels, positions = _extract_groupwise_plotting_data(
            df, grouping_column="treatment", value_column="intensity"
        )
        # data_lists: [[1, 2], [3, 4], [5]]
        # labels: ['A', 'B', 'C']

    Compare multiple columns directly:

    .. code-block:: python

        df = pd.DataFrame({"Protein1": [1, 2, 3], "Protein2": [4, 5, 6], "Protein3": [7, 8, 9]})

        data_lists, labels, positions = _extract_groupwise_plotting_data(
            df, direct_columns=["Protein1", "Protein2", "Protein3"]
        )
        # Each column becomes a group for comparison

    Group with a subgroup column (two conditions per precursor):

    .. code-block:: python

        df = pd.DataFrame(
            {
                "precursor": ["P1", "P1", "P2", "P2"],
                "condition": ["ctrl", "treat", "ctrl", "treat"],
                "intensity": [1.0, 2.0, 3.0, 4.0],
            }
        )

        data_lists, labels, positions = _extract_groupwise_plotting_data(
            df,
            grouping_column="precursor",
            value_column="intensity",
            subgroup_column="condition",
        )
        # labels: ['P1', 'P1', 'P2', 'P2']         (one per cell; dedup in plot for ticks)
        # color_keys: ['ctrl', 'treat', 'ctrl', 'treat']
        # positions: [-0.25, 0.25, 0.75, 1.25]    (with default width=0.5)
    """
    # Handle long vs. wide data input format
    if direct_columns is not None:
        if grouping_column is not None or value_column is not None:
            logger.info("'direct_columns' provided, ignoring 'grouping_column' and 'value_column' parameters.")
        df = data_columns_to_df(data, columns=direct_columns)[direct_columns]  # ensure order
        df = df.melt(var_name="variable", value_name="value")
        grouping_column, value_column = "variable", "value"
    else:
        columns = [grouping_column, value_column]
        if subgroup_column is not None:
            columns.append(subgroup_column)
        df = data_columns_to_df(data, columns=columns)

    # Handle single vs subgrouped plotting
    data_lists: list[list] = []
    labels: list[str] = []
    positions: list[float] = []
    color_keys: list[str] = []

    if subgroup_column is None:
        groups_to_plot = df[grouping_column].dropna().unique().tolist()
        for i, group in enumerate(groups_to_plot):
            group_data = df[df[grouping_column] == group][value_column].dropna()
            if not group_data.empty:
                data_lists.append(group_data.tolist())
                labels.append(group)
                positions.append(float(i))
                color_keys.append(group)
    else:
        main_order = df[grouping_column].dropna().unique().tolist()
        sub_order = df[subgroup_column].dropna().unique().tolist()
        n_sub = len(sub_order)
        # symmetric offsets around the main-group integer position
        offsets = [(j - (n_sub - 1) / 2) * width for j in range(n_sub)]
        for i, group in enumerate(main_order):
            for j, sub in enumerate(sub_order):
                cell = df[(df[grouping_column] == group) & (df[subgroup_column] == sub)][value_column].dropna()
                if cell.empty:
                    continue
                data_lists.append(cell.tolist())
                labels.append(group)
                positions.append(i + offsets[j])
                color_keys.append(sub)

    return data_lists, labels, positions, color_keys


def _set_optional_axis_limits(
    ax: plt.Axes,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Set x and y limits on an Axes object if specified."""
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def add_lines(
    ax: plt.Axes,
    intercepts: float | list[float | int],
    linetype: str = "vline",
    color: str = "black",
    linestyle: str = "--",
    linewidth: float | None = config["linewidths"]["large"],
    line_kwargs: dict | None = None,
) -> None:
    """Add vertical or horizontal reference lines to a plot

    Useful for adding threshold lines to volcano plots, zero lines to bar plots,
    or any other reference lines to visualizations.

    Parameters
    ----------
    ax
        Matplotlib axes object to add lines to
    intercepts
        Single value or list of x-positions (vline) or y-positions (hline)
    linetype
        Type of line: `"vline"` (vertical) or `"hline"` (horizontal)
    color
        Line color
    linestyle
        Line style (e.g., `"--"`, `"-"`, `":"`)
    linewidth
        Line width, defaults to `config["linewidths"]["large"]`
    line_kwargs
        Additional matplotlib line arguments. Note: explicit color, linestyle,
        and linewidth parameters take precedence

    Examples
    --------
    Add significance thresholds to a volcano plot:

    .. code-block:: python

        # Add fold-change thresholds
        add_lines(ax, intercepts=[-1, 1], linetype="vline", color="red", linestyle=":")

        # Add p-value threshold
        add_lines(ax, intercepts=-np.log10(0.05), linetype="hline", color="blue", linestyle="--")

    Add zero reference to bar plot:

    .. code-block:: python

        ax.bar(x, heights)
        add_lines(ax, intercepts=0, linetype="hline", color="black")
    """
    if linetype not in ["vline", "hline"]:
        raise ValueError("linetype must be 'vline' or 'hline'")
    line_func = ax.axvline if linetype == "vline" else ax.axhline

    if not isinstance(intercepts, (list | float | int)):
        raise TypeError("intercepts must be a float, int, or list of floats/ints")

    # handle intercepts and vertical/horizontal lines
    if isinstance(intercepts, float | int):
        intercepts = [intercepts]

    # handle clashes between keyword arguments and line_kwargs
    # explicit parameters take precedence over line_kwargs
    line_kwargs = line_kwargs or {}
    line_kwargs.pop("color", None)
    line_kwargs.pop("linestyle", None)
    line_kwargs.pop("linewidth", None)

    # add lines to ax
    for intercept in intercepts:
        line_func(
            intercept,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            **line_kwargs,
        )


def make_legend_patches(
    color_dict: dict[str, str | tuple],
) -> list[mpl.patches.Patch]:
    """Create colored patches for matplotlib legends

    Converts a label-to-color mapping into matplotlib patches suitable for legends.

    Parameters
    ----------
    color_dict
        Dictionary mapping labels to colors

    Returns
    -------
    list[:class:`matplotlib.patches.Patch`]
        List of colored patches with labels

    Example
    -------
    .. code-block:: python

        # Create patches for categorical legend
        color_dict = {"Control": "blue", "Treatment": "red", "Knockout": "green"}
        patches = make_legend_patches(color_dict)
        ax.legend(handles=patches)
    """
    patches = []
    for value, color in color_dict.items():
        patches.append(
            Patch(
                label=value,
                facecolor=color,
                edgecolor=BaseColors.get("black"),
                linewidth=config["linewidths"]["medium"],
            )
        )

    return patches


def add_legend_to_axes_from_patches(
    ax: plt.Axes,
    patches: list[mpl.patches.Patch],
    **kwargs,
) -> None:
    """Add a legend with patches to an axes, using config defaults for font sizes

    Automatically applies alphapepttools font sizes for legend text and title
    from the config unless overridden.

    Parameters
    ----------
    ax
        Matplotlib axes to add the legend to
    patches
        List of colored patches created by `make_legend_patches`
    **kwargs
        Additional arguments passed to `ax.legend()`.
        If `fontsize` not provided, uses `config["legend"]["font_size"]`

    Example
    -------
    .. code-block:: python

        color_dict = {"WT": "blue", "KO": "red"}
        patches = make_legend_patches(color_dict)
        add_legend_to_axes_from_patches(ax, patches, title="Genotype", loc="upper right")
        # Legend will use config font sizes for text and title
    """
    # create new legend
    if "fontsize" not in kwargs:
        kwargs["fontsize"] = config["legend"]["font_size"]

    _legend = ax.legend(handles=patches, **kwargs)

    # Resize legend title based on config legend title_size
    _legend.set_title(_legend.get_title().get_text(), prop={"size": config["legend"]["title_size"]})


def add_legend_to_axes(
    ax: plt.Axes,
    levels: list[str] | dict[str, str | tuple] | None = None,
    legend: str | mpl.legend.Legend | None = "auto",
    palette: list[str | tuple] | None = None,
    **legend_kwargs,
) -> None:
    """Flexibly add a legend to axes with automatic color assignment

    Handles multiple legend creation patterns: from a list with palette,
    from a color dictionary, or using an existing legend object.
    Automatically switches from qualitative to sequential palette when
    the number of levels exceeds available colors.

    Parameters
    ----------
    ax
        Matplotlib axes to add the legend to
    levels
        Either a list of labels (colors assigned from palette) or
        a dict mapping labels to specific colors
    legend
        `"auto"` creates legend from levels, or pass existing Legend object
    palette
        Custom color palette for list-based levels. If `None`, uses
        qualitative palette (or sequential if too many levels)
    **legend_kwargs
        Additional arguments for legend (title, loc, fontsize, etc.)

    Examples
    --------
    List with automatic colors from palette:

    .. code-block:: python

        # Automatic qualitative palette
        levels = ["Control", "Treatment", "Recovery"]
        add_legend_to_axes(ax, levels=levels, title="Condition")

        # Custom palette
        levels = ["WT", "Het", "KO"]
        palette = ["blue", "lightblue", "red"]
        add_legend_to_axes(ax, levels=levels, palette=palette)

        # Many levels trigger sequential palette
        levels = [f"Sample_{i}" for i in range(20)]
        add_legend_to_axes(ax, levels=levels)  # Switches to sequential

    Dict with explicit color mapping:

    .. code-block:: python

        # Direct color specification
        color_dict = {"Significant": "red", "Not significant": "gray", "Borderline": "orange"}
        add_legend_to_axes(ax, levels=color_dict, title="Status")

    Using existing matplotlib legend:

    .. code-block:: python

        # Pass pre-created legend
        existing_legend = ax.legend(["A", "B"], loc="upper left")
        add_legend_to_axes(other_ax, legend=existing_legend)
    """
    if isinstance(legend, mpl.legend.Legend):
        ax.add_artist(legend)
        return
    if legend == "auto":
        if isinstance(levels, dict):
            patches = make_legend_patches(levels)
            add_legend_to_axes_from_patches(ax, patches, **legend_kwargs)
        elif isinstance(levels, list):
            levels = np.unique(levels)
            if palette is None:
                palette = BasePalettes.get("qualitative")
                if len(levels) > len(palette):
                    palette = BasePalettes.get("sequential")
            color_dict = get_color_mapping(levels, palette)
            patches = make_legend_patches(color_dict)
            add_legend_to_axes_from_patches(ax, patches, **legend_kwargs)
        else:
            logging.warning("No valid 'levels' parameter provided. Skipping legend creation.")
    else:
        logging.warning("No valid 'legend' parameter provided. Skipping legend creation.")


def drop_nan_coordinate_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    labels: np.ndarray | list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove entries where either x or y is NaN, applying same mask to labels

    Filters out instances where either x or y values are NaN, ensuring all three arrays
    (x, y, labels) remain synchronized. This is essential for plotting functions as
    matplotlib cannot handle NaN coordinates. Note that nans in the labels do not cause
    dropping of the respective row, since labels can be strings and missing labels are valid.

    Parameters
    ----------
    x_values
        X-coordinates for plotting.
    y_values
        Y-coordinates for plotting.
    labels
        Labels corresponding to each (x, y) point.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Filtered arrays (x_values, y_values, labels) with NaN rows removed.
        All three arrays maintain the same length and element correspondence.

    Examples
    --------
    >>> x = np.array([1.0, 2.0, np.nan, 4.0])
    >>> y = np.array([5.0, np.nan, 7.0, 8.0])
    >>> labels = np.array(["a", "b", "c", "d"])
    >>> x_clean, y_clean, labels_clean = drop_nan_coordinate_points(x, y, labels)
    >>> x_clean
    array([1., 4.])
    >>> y_clean
    array([5., 8.])
    >>> labels_clean
    array(['a', 'd'], dtype='<U1')

    Notes
    -----
    Uses pandas.isna() to handle both NaN and None values correctly.
    """
    keep_mask = ~(pd.isna(x_values) | pd.isna(y_values))
    return x_values[keep_mask], y_values[keep_mask], labels[keep_mask]


def _get_plot_lims(
    values: np.ndarray,
    padding_factor: float = 1.1,
    sym: str | None = None,
    set_left: float | None = None,
    set_right: float | None = None,
) -> tuple[float, float]:
    """Calculate plot limits with optional symmetry and padding.

    Parameters
    ----------
    values
        Array of values to calculate limits from.
    padding_factor
        Factor to multiply the limits by for padding (e.g., 1.1 for 10% padding).
    sym
        If "max", creates symmetric limits around 0 using the absolute max value.
        If None, uses min and max of values. By default None.
    set_left
        If provided, overrides the calculated left limit with this value. By default None.
    set_right
        If provided, overrides the calculated right limit with this value. By default None.

    Returns
    -------
    tuple[float, float]
        Tuple of (left_limit, right_limit).

    Examples
    --------
    >>> values = np.array([1, 2, 3, -2, -1])
    >>> _get_plot_lims(values, 1.1, sym="max")
    (-3.3, 3.3)
    >>> _get_plot_lims(values, 1.1, set_left=0)
    (0, 3.3)
    """
    # Convert to Series for graceful handling of NaNs
    series = pd.Series(values)

    if sym == "max":
        abs_max = max(abs(series.min()), abs(series.max()))
        left = -abs_max * padding_factor
        right = abs_max * padding_factor
    else:
        left = series.min() * padding_factor
        right = series.max() * padding_factor

    if set_left is not None:
        left = set_left
    if set_right is not None:
        right = set_right

    return (left, right)


def _assign_nearest_anchor_position_to_values(
    values: np.ndarray,
    anchors: list[int | float] | np.ndarray | None,
) -> np.ndarray:
    """Snap values to their nearest anchor positions

    Parameters
    ----------
    values
        Values to be snapped
    anchors
        Anchor positions to snap to. If `None`, returns values unchanged

    Returns
    -------
    :class:`numpy.ndarray`
        Values snapped to nearest anchors

    Example
    -------
    .. code-block:: python

        values = np.array([1.2, 2.7, 5.1])
        anchors = [1, 3, 5]
        result = _assign_nearest_anchor_position_to_values(values, anchors)
        # Returns: [1, 3, 5] - each value snapped to nearest anchor
    """
    if anchors is None:
        return values

    # x-values are binned to the anchor positions
    anchored_values = []

    for val in values:
        anchor_diffs = [abs(anchor - val) for anchor in anchors]
        anchored_values.append(anchors[np.argmin(anchor_diffs)])

    return np.array(anchored_values)


def label_plot(
    ax: plt.Axes,
    data: pd.DataFrame | ad.AnnData,
    x_column: str,
    y_column: str,
    label_column: str,
    x_anchors: list[int | float] | np.ndarray | None = None,
    label_kwargs: dict | None = None,
    line_kwargs: dict | None = None,
    label_parser: Callable | None = None,
    y_display_start: float = 1,
    y_padding_factor: float = 3,
) -> None:
    """Add labels to a 2D axes object

    Add labels to a plot based on x and y coordinates. The labels are either placed near the datapoint
    or anchored to the left or right of the plot, where labels below the splitpoint are anchored to the
    left and labels above the splitpoint are anchored to the right.

    Parameters
    ----------
    ax
        Matplotlib axes object to add the labels to.
    x_values
        x-coordinates of the labels.
    y_values
        y-coordinates of the labels.
    labels
        Labels to add to the plot.
    x_anchors
        x-coordinates of the anchors to use for the labels. If None, labels are placed at the x-coordinates of the data points. By default None.
    label_kwargs
        Additional keyword arguments for the label text, by default None.
    line_kwargs
        Additional keyword arguments for the line connecting the label to the data point, by default None.
    label_parser
        Function to parse the labels, by default None. This is useful to convert
        labels from a computation-context to presentation context, e.g. a column
        like upregulated_proteins could be shown as "Upregulated Proteins" in the plot.
    y_display_start : float, optional
        Starting point for the y-coordinates of the labels, by default 1.
    y_padding_factor: float, optional
        Factor to increase or decrease how far apart labels are spread in the y-direction when stacked into a column over x-anchors

    Returns
    -------
    None
        The function modifies the axes in place and does not return anything.

    Examples
    --------
    Basic scatter plot with labels:

    .. code-block:: python

        import pandas as pd
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt
        from alphapepttools.pl import label_plot

        # Sample data
        df = pd.DataFrame({"x": [-2, -1, 0, 1, 2], "y": [3, 5, 2, 6, 4], "label": ["A", "B", "C", "D", "E"]})

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.scatter(ax=ax, data=df, x_column="x", y_column="y")
        label_plot(ax, df["x"], df["y"], df["label"])

    With anchored labels on left and right sides:

    .. code-block:: python

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.scatter(ax=ax, data=df, x_column="x", y_column="y")
        label_plot(ax, df["x"], df["y"], df["label"], x_anchors=[-2.5, 2.5], y_display_start=4, y_padding_factor=10)

    """
    label_kwargs = {"fontsize": config["font_sizes"]["medium"], **(label_kwargs or {})}
    line_kwargs = {"color": BaseColors.get("black"), "linewidth": config["linewidths"]["medium"], **(line_kwargs or {})}
    label_parser = label_parser or (lambda x: x)

    # Extract all needed columns into a DataFrame
    df = data_columns_to_df(data, columns=[x_column, y_column, label_column])

    # Sort by y values (highest to lowest)
    df = df.sort_values(by=y_column, ascending=False)

    # convert to numpy arrays for consistency & remove any nans
    x_values = data_column_to_array(df, x_column)
    y_values = data_column_to_array(df, y_column)
    labels = data_column_to_array(df, label_column)

    # Remove any nans
    x_values, y_values, labels = drop_nan_coordinate_points(x_values, y_values, labels)

    # determine label positions based on optional x_anchors
    if x_anchors is not None:
        # x-values are binned to the anchor positions
        anchored_x_values = _assign_nearest_anchor_position_to_values(x_values, x_anchors)

        # y-values should be distributed evenly between the min and max y-values at that anchor
        label_spacing_display = config["font_sizes"]["medium"] * y_padding_factor

        # Translate label spacing from display coordinates to axes coordinates, since the same spacing should appear regardless of y-values
        transform = ax.transData.inverted()
        _, y_spacing_in_data_coords = transform.transform((0, label_spacing_display)) - transform.transform((0, 0))

        # get a consistent starting point for y values with respect to the actual display window
        _, upper_bound_in_data_coords = transform.transform((0, ax.get_window_extent().height * y_display_start))

        # Iterate over all unique x_anchors and assign y-values in data coordinates to the respective labels
        # TODO: Optimize this loop to not have so many data structures
        sorted_labels = []
        sorted_data_x_values = []
        sorted_data_y_values = []
        sorted_label_x_values = []
        sorted_label_y_values = []

        for anchor_value in np.unique(anchored_x_values):
            # Get the sequence of sorted values for the current anchor
            anchor_mask = anchored_x_values == anchor_value

            sorted_labels.extend(list(labels[anchor_mask]))
            sorted_data_x_values.extend(list(x_values[anchor_mask]))
            sorted_data_y_values.extend(list(y_values[anchor_mask]))
            sorted_label_x_values.extend(list([anchor_value] * np.sum(anchor_mask)))
            sorted_label_y_values.extend(
                [upper_bound_in_data_coords - y_spacing_in_data_coords * i for i in range(np.sum(anchor_mask))]
            )

    else:
        sorted_labels = labels
        sorted_data_x_values = x_values
        sorted_data_y_values = y_values
        sorted_label_x_values = x_values
        sorted_label_y_values = y_values

    # generate lines from data values to label positions
    lines = []
    for label, x, y, label_x, label_y in zip(
        sorted_labels,
        sorted_data_x_values,
        sorted_data_y_values,
        sorted_label_x_values,
        sorted_label_y_values,
        strict=True,
    ):
        lines.append(((x, label_x), (y, label_y), label))

    for line in lines:
        ax.plot(line[0], line[1], **line_kwargs)
        if x_anchors is not None:
            alignment = "right" if line[0][0] > line[0][1] else "left"
            label_kwargs["ha"] = alignment
        ax.text(line[0][1], line[1][1], label_parser(line[2]), **label_kwargs)


def _array_to_str(
    array: np.ndarray | pd.Series,
) -> np.ndarray:
    """Map a numpy array to string values."""
    return np.array(array, dtype=object).astype(str)


def _dict_keys_to_str(
    dictionary: dict,
) -> dict[str, Any]:
    """Convert the keys of a dictionary to strings."""
    return {str(k): v for k, v in dictionary.items()}


@dataclass(frozen=True)
class PlotConfig:
    """Base configuration for all plot types

    This dataclass holds common configuration parameters for different plot types.
    It allows handling of different arguments for different plot types, eliminating the need to
    repeat shared parameters across different instances of the same plot.

    Usage
    -----



    """

    data: ad.AnnData | pd.DataFrame | None = None
    _extra: dict[str, Any] | None = None

    def __post_init__(self):
        # Initialize _extra as empty dict if None
        if self._extra is None:
            object.__setattr__(self, "_extra", {})

    def __getattr__(self, name: str) -> object:
        """Access extra fields as attributes"""
        if name in self._extra:
            return self._extra[name]
        raise KeyError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def copy_with(
        self,
        **changes: dict[str, Any],
    ) -> "PlotConfig":
        """Create a copy with specified changes"""
        # Make it so that changes can override existing data & _extra fields
        current = {"data": self.data}
        current.update(self._extra)
        current.update(changes)

        # Since PlotConfig is instantiated with data and _extra, separate them here
        data = current.pop("data", None)
        new_extra = current  # Everything else goes into _extra

        return PlotConfig(data=data, _extra=new_extra)

    def to_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs dict"""
        result = {}
        if self.data is not None:
            result["data"] = self.data
        result.update(self._extra)
        return {k: v for k, v in result.items() if v is not None}


def make_scatter_config(data: ad.AnnData | pd.DataFrame, x_column: str, y_column: str, **kwargs) -> PlotConfig:
    """Create a config for scatter plots

    Parameters
    ----------
    data : ad.AnnData | pd.DataFrame
        Data to plot
    x_column : str
        Column for x-axis
    y_column : str
        Column for y-axis
    **kwargs
        Additional scatter parameters (color, scatter_kwargs, etc.)

    Returns
    -------
    PlotConfig
        Configuration object with all parameters set
    """
    extra = {"x_column": x_column, "y_column": y_column}
    extra.update(kwargs)
    return PlotConfig(data=data, _extra=extra)


def _extract_plot_layer_specs(layer_specs: tuple) -> tuple[str, str | int | list, str, dict]:
    """Extract (column, value, color_key[, kwargs]) from layer specification.

    Parameters
    ----------
    layer_specs : tuple
        Layer specification with 3-4 elements:
        - layer_column (str): Column name to filter on
        - layer_val (str|int|list): Value(s) to match
        - color_key (str): Key to lookup in color_dict
        - scatter_kwargs (dict, optional): Additional scatter parameters

    Returns
    -------
    tuple
        (layer_column, layer_val, color_key, scatter_kwargs)
    """
    min_required_elements = 3
    max_allowed_elements = 4

    if len(layer_specs) < min_required_elements:
        raise ValueError(f"Layer specs must have at least {min_required_elements} elements, got {len(layer_specs)}")
    if len(layer_specs) > max_allowed_elements:
        raise ValueError(f"Layer specs must have at most {max_allowed_elements} elements, got {len(layer_specs)}")

    layer_column, layer_val, color_key = layer_specs[:min_required_elements]

    if not isinstance(layer_column, str):
        raise TypeError(f"layer_column must be str, got {type(layer_column).__name__}")
    if not isinstance(layer_val, (str, int, list)):
        raise TypeError(f"layer_val must be str|int|list, got {type(layer_val).__name__}")
    if not isinstance(color_key, str):
        raise TypeError(f"color_key must be str, got {type(color_key).__name__}")

    # Get kwargs from the last element if present & validate type
    scatter_kwargs = {}
    if len(layer_specs) == max_allowed_elements:
        scatter_kwargs = layer_specs[min_required_elements]
        if not isinstance(scatter_kwargs, dict):
            raise TypeError(f"scatter_kwargs must be dict, got {type(scatter_kwargs).__name__}")

    return layer_column, layer_val, color_key, scatter_kwargs


_REMOVED_PLOTS_METHODS = (
    "layered_plot",
    "histogram",
    "scatter",
    "barplot",
    "boxplot",
    "violinplot",
    "rank_median_plot",
    "plot_pca",
    "scree_plot",
    "plot_pca_loadings",
    "plot_pca_loadings_2d",
    "volcano",
)


class _PlotsRemovedMeta(type):
    def __getattr__(cls, name: str):
        if name not in _REMOVED_PLOTS_METHODS:
            raise AttributeError(f"type object 'Plots' has no attribute {name!r}")

        def _removed(*args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError(
                f"The 'Plots' wrapper class was removed, "
                f"please update the call from 'Plots.{name}(...)' "
                f"to 'alphapepttools.pl.{name}(...)'. \nIf you loaded alphapepttools as apt, then 'apt.pl.{name}(...)' is equivalent."
            )

        return _removed


class Plots(metaclass=_PlotsRemovedMeta):
    """Removed. Methods are now standalone functions in ``alphapepttools.pl``.

    Migration: replace ``Plots.<name>(...)`` with ``alphapepttools.pl.<name>(...)``. If you loaded alphapepttools as apt, then ``apt.pl.<name>(...)`` is equivalent.
    """

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "The 'Plots' wrapper class was removed. Use the standalone "
            "functions in alphapepttools.pl (e.g. alphapepttools.pl.scatter(...)) directly. "
            "If you loaded alphapepttools as apt, then apt.pl.scatter(...) is equivalent."
        )


def layered_plot(
    ax: plt.Axes,
    base_config: PlotConfig,
    layers: list[tuple] | None = None,
    color_dict: dict[str, str | tuple] | None = None,
    default_layer_column: str = "__data",
    default_layer_val: str = "__all",
    default_color_key: str = "__default_color",
    default_color: str | tuple = BaseColors.get("blue"),
    default_layer_kwargs: dict | None = None,
    lim_padding_factor: float = 1.1,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    plotting_callable: Callable | None = None,
    return_glob_layer_indices: bool = False,  # noqa: FBT001, FBT002
) -> None | list:
    """Plot multiple layers with defined hierarchy and without datapoint reuse.

    In order to layer multiple levels of plotting layers onto each other (e.g. color points by differential
    expression, and then highlight extra points on top of that), this function allows for defining multiple
    plotting layers.
    The purpose of this is to avoid repetitive specification of shared parameters in the layers list. Points
    which are not assigned to any layer are plotted in the default layer in the background. A shared color_dict
    is used by all layers to lookup colors by key.

    Parameters
    ----------
    ax
        Matplotlib axes object to plot on.
    base_config
        Base configuration for the plot layers containing data and plot parameters.
    layers
        List of layer specifications, each a tuple of (layer_column, layer_val, color_key, layer_kwargs).
        - layer_column (str): Column name to filter on for this layer
        - layer_val (str|int|list): Value(s) to match in the layer_column (i.e. this layer will only contain points
        where data[layer_column] is in layer_val)
        - color_key (str): Color key in color_dict for this layer
        - layer_kwargs (dict, optional): Additional parameters for this layer
        By default None, which results in a single default layer containing all points.
    color_dict
        Dictionary mapping color keys to colors. By default None, which results in a single default color.
    default_layer_column
        Column name to use for the default layer. By default "__data".
    default_layer_val
        Value to use for the default layer. By default "__all".
    default_color_key
        Color key to use for the default color. By default "__default_color".
    default_color
        Color to use for the default color. By default BaseColors.get("blue").
    default_layer_kwargs
        Default scatterplot keyword arguments for the default layer. By default None.
    lim_padding_factor
        Factor to pad the x and y limits of the plot. By default 1.1.
    xlims
        x-axis limits for the plot. If None, limits are calculated from the data with padding. By default None.
    ylims
        y-axis limits for the plot. If None, limits are calculated from the data with padding. By default None.
    plotting_callable
        Custom plotting function to use instead of scatter. Should accept ax, data, and other parameters.
        By default None, which uses scatter.
    return_glob_layer_indices
        If True, returns a list of (indices, color, color_key, scatter_kwargs) tuples for each layer.
        Useful for debugging or further processing of layer assignments. By default False.

    Returns
    -------
    None | list
        If return_glob_layer_indices is False, returns None. Otherwise, returns a list of tuples
        containing (layer_indices, layer_color, color_key, scatter_kwargs) for each layer.

    Example
    -------
    .. code-block:: python

        import numpy as np
        import pandas as pd
        import anndata as ad
        from alphapepttools import pl
        import alphapepttools as apt
        from alphapepttools.pl import BaseColors, create_figure, label_axes

        rng = np.random.default_rng(seed=42)

        # example data
        testx = rng.normal(0, 1, 300)
        testy = -np.cos(testx) + rng.normal(0, 0.2, 300)
        testp = 10 ** -(testy - min(testy))
        vp_data = pd.DataFrame(
            {
                "id": [f"P{10000 + i}" for i in range(300)],
                "gene": [f"gene_{i}" for i in range(300)],
                "log2fc": testx,
                "pval": testp,
                "neg_log10pval": -np.log10(testp),
            }
        )
        vp_data.index = vp_data["id"].astype(str)

        example_adata_diff = ad.AnnData(
            X=vp_data[["log2fc", "pval", "neg_log10pval"]].values,
            obs=vp_data[["id", "gene"]],
            var=vp_data[["log2fc", "pval", "neg_log10pval"]].columns.to_frame(),
        )

        # Add some example categorical and point-of-interest annotations
        example_adata_diff.obs["diff_exp_status"] = example_adata_diff.to_df()["log2fc"].apply(
            lambda x: "upregulated" if x > 1 else ("downregulated" if x < -1 else "unchanged")
        )

        example_adata_diff.obs["pathway"] = rng.choice(
            ["pathway_A", "pathway_B", "pathway_C", "pathway_D", "pathway_E"], size=example_adata_diff.n_obs
        )
        example_adata_diff.obs["poi_status"] = rng.choice(
            ["poi", "background"], size=example_adata_diff.n_obs, p=[0.01, 0.99]
        )

        # Specify a custom color dictionary for the categories we want to color
        color_dict = {
            "upregulated": BaseColors.get("red"),
            "downregulated": BaseColors.get("blue"),
            "unchanged": BaseColors.get("grey"),
            "poi": BaseColors.get("black"),
            "pathway_A": BaseColors.get("purple", lighten=0.5),
        }

        # In order to avoid repeating instructions for each layer, we can summarize the parameters for the whole plot in a configuration
        layered_plot_config = pl.make_scatter_config(
            data=example_adata_diff,  # all layers use the same data and numeric columns
            x_column="log2fc",
            y_column="neg_log10pval",
            scatter_kwargs={"alpha": 0.7, "s": 30},  # make the points slightly transparent and set a good size
        )

        # Define layers: each layer is a tuple of (column to filter on, value to select, color key, optional plotting kwargs)
        plot_layers = [
            (
                "poi_status",
                "poi",
                "poi",
                {"scatter_kwargs": {"marker": "^", "s": 200}},
            ),  # this layer gets custom scatterplot settings kwargs
            ("pathway", "pathway_A", "pathway_A"),
            ("diff_exp_status", "upregulated", "upregulated"),
            ("diff_exp_status", "downregulated", "downregulated"),
            ("diff_exp_status", "unchanged", "unchanged"),
        ]

        # Visualize the plot with layers
        fig, axm = create_figure(1, 1, figsize=(6, 6))
        ax = axm.next()
        apt.pl.layered_plot(
            ax=ax,
            base_config=layered_plot_config,
            layers=plot_layers,
            color_dict=color_dict,
            ylims=(0, 3),
            xlims=(-3, 4.5),  # leave some space for labels on the right side
        )

        # Label axes
        label_axes(
            ax,
            xlabel="log2 Fold Change",
            ylabel="-log10 p-value",
            title="Layered Volcano Plot of Differential Expression with Custom Colors and Highlighting",
        )

    """
    color_dict = color_dict or {default_color_key: default_color}
    default_layer_kwargs = default_layer_kwargs or {}
    base_config = base_config or PlotConfig(data=None)
    plotting_callable = plotting_callable or scatter

    # Get data from base_config
    data = base_config.data
    x_column = base_config.x_column
    y_column = base_config.y_column

    # By default, all datapoints are in the default layer
    if layers is None:
        data[default_layer_column] = data_index_to_array(data, "obs")
        data[default_layer_column] = default_layer_val

    layers = layers or [(default_layer_column, default_layer_val, default_color_key, default_layer_kwargs)]

    # We need to ensure that we have consistent limits across all layers
    xlims = xlims or _get_plot_lims(data_column_to_array(data, x_column), lim_padding_factor, sym="max")
    ylims = ylims or _get_plot_lims(data_column_to_array(data, y_column), lim_padding_factor, sym="max")

    # Prior to plotting, gather indices for each layer, ensuring no datapoint is used twice
    glob_spent_idxs = []
    glob_layer_idxs = []
    entry_indices = np.arange(len(data))
    for layer_specs in layers:
        # Flexibly extract layer specifications: layer_kwargs are optional
        layer_column, layer_val, color_key, layer_kwargs = _extract_plot_layer_specs(layer_specs)

        # Create index mask for current layer
        layer_column_array = data_column_to_array(data, layer_column)
        current_layer_mask = np.isin(layer_column_array, _tolist(layer_val))

        # Update the current layer mask to exclude points already assigned to previous layers
        not_spent = ~np.isin(entry_indices, glob_spent_idxs)
        current_layer_mask = current_layer_mask & not_spent

        # Save indices for current layer
        layer_idxs = entry_indices[current_layer_mask].tolist()

        # Lookup color for current layer
        layer_color = color_dict.get(color_key, default_color)
        glob_layer_idxs.append((layer_idxs, layer_color, color_key, layer_kwargs))

        # Update spent indices so they are not assigned again
        glob_spent_idxs.extend(layer_idxs)

    # If any indices are not spent, assign them to the default layer
    if len(glob_spent_idxs) < len(data):
        remaining_idxs = list(set(entry_indices) - set(glob_spent_idxs))
        glob_layer_idxs.append(
            (
                remaining_idxs,
                color_dict.get(default_color_key, default_color),
                default_color_key,
                default_layer_kwargs,
            )
        )
        glob_spent_idxs.extend(remaining_idxs)

    # Check that no points were left unassigned
    if len(glob_spent_idxs) != len(data):
        raise ValueError("Some data points were not assigned to any layer in the volcano plot.")

    # Plot each layer in reverse order so that the first layer is on top
    for layer_idxs, layer_color, _, layer_kwargs in reversed(glob_layer_idxs):
        if len(layer_idxs) > 0:
            # Global plotting parameters are retained/updated from the base_config.
            layer_config = base_config.copy_with(
                data=subset_data(data, layer_idxs),
                color=layer_color,
                **layer_kwargs,
            )
            # ax and limits must be passed explicitly
            plotting_callable(ax=ax, xlim=xlims, ylim=ylims, **layer_config.to_kwargs())

    if return_glob_layer_indices:
        return glob_layer_idxs

    return None


def histogram(
    data: pd.DataFrame | ad.AnnData,
    value_column: str,
    color_map_column: str | None = None,
    bins: int = 10,
    ax: plt.Axes | None = None,
    color: str = "blue",
    palette: list[tuple] | None = None,
    color_dict: dict[str, str | tuple] | None = None,
    legend: str | mpl.legend.Legend | None = None,
    hist_kwargs: dict | None = None,
    legend_kwargs: dict | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot a histogram from a DataFrame or AnnData object

    Creates a histogram showing the distribution of values, with optional
    grouping by a categorical column. When grouping is used, overlapping
    histograms are created with the same bin edges for easy comparison.

    Parameters
    ----------
    data
        Data to plot, must contain the value_column and optionally
        the color_map_column for grouping.
    value_column
        Column containing numeric values to plot in the histogram.
    color_map_column
        Column for categorical grouping. Each unique value gets its own
        colored histogram overlay. NaN values are converted to strings.
    bins
        Number of bins for the histogram. Default is 10.
    ax
        Matplotlib axes to plot on. If None, a new figure is created.
    color
        Single color for ungrouped histogram. Default is "blue".
    palette
        Color palette for grouped histograms. Defaults to qualitative palette.
    color_dict
        Explicit mapping of groups to colors. Overrides palette if provided.
    legend
        If "auto", creates legend for grouped data. Can also pass existing Legend.
    hist_kwargs
        Additional arguments for matplotlib.hist() like:
        - alpha: transparency (0-1)
        - histtype: 'bar', 'step', 'stepfilled'
        - edgecolor: outline color
        - linewidth: outline width
    legend_kwargs
        Additional arguments for legend like title, loc, fontsize.
    xlim
        X-axis limits as (min, max).
    ylim
        Y-axis limits as (min, max).

    Returns
    -------
    None

    Examples
    --------
    Simple histogram:

    .. code-block:: python

        import pandas as pd
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt

        df = pd.DataFrame({"intensity": [1.5, 2.3, 2.8, 1.9, 3.1, 2.5]})

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.histogram(data=df, value_column="intensity", bins=30, color="skyblue", ax=ax)

    Grouped histogram with transparency:

    .. code-block:: python

        import pandas as pd
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt

        df = pd.DataFrame(
            {
                "values": [1.5, 2.3, 2.8, 1.9, 3.1, 2.5, 4.2, 3.8],
                "condition": ["A", "A", "B", "B", "A", "B", "A", "B"],
            }
        )

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.histogram(
            data=df,
            value_column="values",
            color_map_column="condition",
            bins=20,
            legend="auto",
            hist_kwargs={"alpha": 0.7, "histtype": "stepfilled"},
            legend_kwargs={"title": "Condition"},
            ax=ax,
        )

    Custom color mapping:

    .. code-block:: python

        import pandas as pd
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt

        example_df = pd.DataFrame(
            {
                "values": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "levels": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            }
        )

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.histogram(
            data=example_df,
            value_column="values",
            color_map_column="levels",
            color_dict={"A": "red", "B": "blue", "C": "green"},
            bins=20,
            ax=ax,
            legend="auto",
            hist_kwargs={"alpha": 0.7, "histtype": "stepfilled", "edgecolor": "k"},
            legend_kwargs={"title": "Levels", "loc": "upper left"},
        )

    Notes
    -----
    - When grouping data, all groups use the same bin edges for comparison
    - Unmapped groups in color_dict default to grey
    - NaN values are excluded from the histogram
    """
    hist_kwargs = hist_kwargs or {}
    legend_kwargs = legend_kwargs or {}

    if ax is None:
        _, ax = create_figure(1, 1)

    values = data_column_to_array(data, value_column)

    if color_map_column is None:
        color = BaseColors.get(color)
        ax.hist(values, bins=bins, color=color, **hist_kwargs)
    else:
        color_levels = _array_to_str(data_column_to_array(data, color_map_column))
        color_dict = _dict_keys_to_str(
            color_dict or get_color_mapping(color_levels, palette or BasePalettes.get("qualitative"))
        )

        for level in set(color_levels) - set(color_dict):
            color_dict[level] = BaseColors.get("grey")

        # Calculate unified bin edges based on the entire data range
        values_clean = values[~np.isnan(values)]
        data_min = np.min(values_clean)
        data_max = np.max(values_clean)

        # Create unified bin edges for the entire data range
        unified_bin_edges = np.linspace(data_min, data_max, bins + 1)

        for level, level_color in color_dict.items():
            level_values = values[color_levels == level]
            level_values = level_values[~np.isnan(level_values)]

            if len(level_values) == 0:
                continue

            # Use the unified bin edges for all sub-histograms
            ax.hist(
                level_values,
                bins=unified_bin_edges,
                color=level_color,
                **hist_kwargs,
            )

        if legend is not None:
            add_legend_to_axes(
                ax=ax,
                levels=color_dict,
                legend=legend,
                **legend_kwargs,
            )

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def scatter(
    data: ad.AnnData | pd.DataFrame,
    x_column: str,
    y_column: str,
    color: str | None = None,
    color_map_column: str | None = None,
    color_column: str | None = None,
    ax: plt.Axes | None = None,
    palette: list[str | tuple] | None = None,
    color_dict: dict[str, str | tuple] | None = None,
    legend: str | mpl.legend.Legend | None = None,
    scatter_kwargs: dict | None = None,
    legend_kwargs: dict | None = None,
    figure_kwargs: dict | None = None,
    default_group: str = "__data",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    order: Literal["color_frequency", "original"] = "color_frequency",
) -> None:
    """Plot a scatterplot from a DataFrame or AnnData object

    Coloring works in three ways, with the following order of precedence: 1. color_column, 2. color_map_column, 3. color.
    If a color_column is provided, its values are interpreted directly as colors, i.e. they have to be something matplotlib
    can understand (e.g. RGBA, hex, etc.). If a color_map_column is provided, its values are mapped to colors in combination
    with palette or color_dict (see color mapping logic below). If neither color_column nor color_map_column is provided, the
    color parameter is used to color all points the same (defaults to blue).

    Color mapping logic
    -------------------
    - color_map_column is non-numeric:
        - If color_dict is not None: Use color_dict to assign levels of color_map_column to colors (unmapped levels default to grey).
        - If color_dict is None, and palette is not None: Use palette to automatically assign colors to each level.
        - If color_dict is None and palette is None: Use a repeating default palette to assign colors to each level.
    - color_map_column is numeric:
        - If palette is a matplotlib colormap: Quantitatively map values to colors using the colormap. This means that e.g. 1 and 3 will be closer in color than 1 and 10.
        - If palette is not a matplotlib colormap: Treat numeric values as categorical and color as described above.

    Parameters
    ----------
    data
        Data to plot, must contain the x_column and y_column and optionally the color_column or color_map_column.
    x_column
        Column in data to plot on the x-axis. Must contain numeric data.
    y_column
        Column in data to plot on the y-axis. Must contain numeric data.
    color
        Color to use for the scatterplot. By default "blue".
    color_map_column
        Column in data to use for color encoding. These values are mapped to the palette or the color_dict (see below).
        Its values cannot contain NaNs, therefore color_map_column is coerced to string and missing values replaced by
        a default filler string. Overrides color parameter. By default None.
    color_column
        Column in data to plot the colors. This must contain actual color values (RGBA, hex, etc.).
        Overrides color and color_map_column parameters. By default None.
    ax
        Matplotlib axes object to plot on, if None a new figure is created. By default None.
    palette
        List of colors to use for color encoding, if None a default palette is used.
        Can be a matplotlib Colormap for continuous gradients. By default None.
    color_dict
        Supercedes palette, a dictionary mapping levels to colors. By default None. If provided, palette is ignored.
    legend
        Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
    scatter_kwargs
        Additional keyword arguments for the matplotlib scatter function (s, alpha, edgecolors, etc.). By default None.
    legend_kwargs
        Additional keyword arguments for the matplotlib legend function. By default None.
    figure_kwargs : dict | None, optional
        Additional keyword arguments for figure creation. By default None.
    xlim
        Limits for the x-axis. By default None.
    ylim
        Limits for the y-axis. By default None.
    order : str
        Ordering of plotting data points. If "color_frequency", the rarest occuring colors are plotted on top. This is the default
        and follows the assumption that rarer categories are more important to the plot's message (e.g. 1000 grey points should not cover 100 green points, which should not cover 10 red points).
        If "original", the order of the data is kept as is, which is useful for plotting ordered categorical datapoints.

    Returns
    -------
    None

    Examples
    --------
    Simple scatter with single color:

    .. code-block:: python

        import pandas as pd
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt

        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 1, 3, 5]})

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.scatter(data=df, x_column="x", y_column="y", color="red", ax=ax)

    Categorical coloring with automatic palette:

    .. code-block:: python

        import pandas as pd
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt

        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 1, 3, 5],
                "category": ["A", "B", "A", "C", "B"],
            }
        )

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.scatter(
            data=df,
            x_column="x",
            y_column="y",
            color_map_column="category",
            legend="auto",
            ax=ax,
        )

    Custom color dictionary:

    .. code-block:: python

        import pandas as pd
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt

        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 1, 3, 5],
                "significance": ["significant", "not_significant", "significant", "not_significant", "significant"],
            }
        )

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.scatter(
            data=df,
            x_column="x",
            y_column="y",
            color_map_column="significance",
            color_dict={"significant": "red", "not_significant": "gray"},
            legend="auto",
            scatter_kwargs={"s": 50, "alpha": 0.7},
            ax=ax,
        )

    Quantitative gradient with numeric data:

    .. code-block:: python

        import pandas as pd
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt
        from alphapepttools.pl.colors import BaseColormaps

        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 1, 3, 5],
                "intensity": [1.0, 5.0, 10.0, 15.0, 20.0],
            }
        )

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.scatter(
            data=df,
            x_column="x",
            y_column="y",
            color_map_column="intensity",
            palette=BaseColormaps.get("sequential"),
            ax=ax,
        )

    Direct color values from column:

    .. code-block:: python

        import pandas as pd
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt

        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 1, 3, 5],
                "my_colors": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"],
            }
        )

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.scatter(
            data=df,
            x_column="x",
            y_column="y",
            color_column="my_colors",
            ax=ax,
        )

    Notes
    -----
    - Points are ordered by color frequency (most frequent in back) for better visibility
    - Unmapped values in color_dict default to grey
    - NaN values in color columns are handled as strings
    """
    scatter_kwargs = scatter_kwargs or {}
    legend_kwargs = legend_kwargs or {}
    figure_kwargs = figure_kwargs or {"figsize": (3, 3)}

    default_color = BaseColors.get("blue")

    if ax is None:
        _, axm = create_figure(**figure_kwargs)
        ax = axm.next()

    # Directly use colors from the color_column
    if color_column is not None:
        color_values = data_column_to_array(data, color_column)
    # Map values from the color_map_column to colors
    elif color_map_column is not None:
        color_map_column_array = data_column_to_array(data, color_map_column)

        if pd.api.types.is_numeric_dtype(color_map_column_array) and isinstance(palette, plt.Colormap):
            color_values = _get_colors_from_cmap(
                cmap_name=palette,
                values=color_map_column_array,
            )
        # if color_map_column is not numeric
        else:
            color_map_column_array = _array_to_str(data_column_to_array(data, color_map_column))
            color_dict = _dict_keys_to_str(
                color_dict
                or get_color_mapping(values=color_map_column_array, palette=palette or BasePalettes.get("qualitative"))
            )

            for level in set(color_map_column_array) - set(color_dict):
                color_dict[level] = BaseColors.get("grey")

            color_values = np.array([color_dict[level] for level in color_map_column_array], dtype=object)
    else:
        color_dict = {default_group: color or default_color}
        color_values = np.array([color_dict[default_group]] * len(data))

    # Get base arrays
    x_values = data_column_to_array(data, x_column)
    y_values = data_column_to_array(data, y_column)
    color_values = np.array(color_values)

    # Order points by color frequency if needed, so that points that occur only rarely are plotted on top.
    # This solves issues with e.g. plotting 1000 points and coloring 10 of them red, where presumable the red ones should overplot the grey ones but not vice versa.
    if order == "color_frequency":
        counts = Counter([str(cv) for cv in color_values])
        order_indices = np.argsort([counts[str(cv)] for cv in color_values])[::-1]

        x_values = x_values[order_indices]
        y_values = y_values[order_indices]
        color_values = color_values[order_indices]

        # In case users pass an array-like in kwargs, make sure the order is consistent. This concerns e.g. edgecolor, size, etc.
        iterable_kwargs = find_iterable_kwargs(scatter_kwargs, match_length=len(color_values))
        for k, v in iterable_kwargs.items():
            scatter_kwargs[k] = list(np.array(v)[order_indices])

    ax.scatter(
        x=x_values,
        y=y_values,
        c=color_values,
        **scatter_kwargs,
    )

    if legend is not None and color_dict is not None:
        add_legend_to_axes(
            ax=ax,
            levels=color_dict,
            legend=legend,
            **legend_kwargs,
        )

    # TODO: set this to flexible limits with optional symmetric padding
    _set_optional_axis_limits(
        ax=ax,
        xlim=xlim,
        ylim=ylim,
    )


def barplot(
    ax: plt.Axes,
    data: ad.AnnData | pd.DataFrame,
    grouping_column: list[str] | None = None,
    value_column: list[str] | None = None,
    direct_columns: list[str] | None = None,
    color: tuple = BaseColors.get("blue"),
    color_dict: dict | None = None,
    subgroup_column: str | None = None,
    width: float = 0.4,
    legend: str | mpl.legend.Legend | None = None,
    legend_kwargs: dict | None = None,
) -> None:
    """Plot a bar chart from a DataFrame or AnnData object

    Creates a bar plot showing means with error bars (standard deviation) for grouped data.
    Each bar represents the mean of values within a group, with error bars showing the
    standard deviation. Bars have semi-transparent fill with opaque black outlines.

    Two modes of operation:
    1. **Grouping mode**: Use grouping_column/value_column to group data by categories
    2. **Direct mode**: Use direct_columns to compare multiple columns directly

    Parameters
    ----------
    ax
        Matplotlib axes object to plot on.
    data
        Data containing grouping and value columns or direct columns to plot.
    grouping_column
        Column containing the groups to compare (categorical).
        Used with value_column for grouped comparisons. By default None.
    value_column
        Column whose values should be plotted (numeric).
        Used with grouping_column for grouped comparisons. By default None.
    direct_columns
        List of column names to compare directly. Each column becomes a separate bar.
        Overrides grouping_column and value_column. By default None.
    color
        Default color for all bars. By default BaseColors.get("blue").
    color_dict
        Dictionary mapping group labels to specific colors. Overrides the color
        parameter for specified groups. By default None.

    Returns
    -------
    None

    Examples
    --------
    Grouped comparison (long format):

    .. code-block:: python

        import pandas as pd
        import anndata as ad
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt

        data = pd.DataFrame({"intensity": [1, 2, 3, 4, 5, 6, 7]})
        obs = pd.DataFrame({"group": ["A", "A", "B", "B", "B", "C", "C"]})
        adata = ad.AnnData(X=data.values, obs=obs, var=pd.DataFrame(index=data.columns))

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.barplot(
            ax=ax,
            data=adata,
            grouping_column="group",
            value_column="intensity",
            color_dict={"A": "red", "B": "green", "C": "blue"},
        )

    Direct column comparison (wide format):

    .. code-block:: python

    Notes
    -----
    - Error bars show standard deviation of values within each group
    - Bars have 50% transparency with opaque black outlines
    - When using direct_columns, each column's mean is calculated across all rows
    - Missing values (NaN) are excluded from mean and std calculations
    """
    data, labels, positions, color_keys = _extract_groupwise_plotting_data(
        data=data,
        grouping_column=grouping_column,
        value_column=value_column,
        direct_columns=direct_columns,
        subgroup_column=subgroup_column,
        width=width,
    )

    means = [pd.Series(d).mean() for d in data]
    stds = [pd.Series(d).std() for d in data]

    bars = ax.bar(
        x=positions,
        height=means,
        yerr=stds,
        capsize=5,
        align="center",
        width=width,
    )

    # Styling of bars
    for color_key, bar in zip(color_keys, bars, strict=False):
        current_color = color_dict.get(color_key, config["na_color"]) if color_dict else color
        bar.set_facecolor(mcolors.to_rgba(current_color, alpha=0.5))
        bar.set_edgecolor(BaseColors.get("black"))
        bar.set(linewidth=config["linewidths"]["medium"])

    # One label per position in case subgroups returned repeated entries
    unique_labels = list(dict.fromkeys(labels))

    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels)

    legend_kwargs = legend_kwargs or {}
    if legend is not None and color_dict is not None:
        add_legend_to_axes(
            ax=ax,
            levels=color_dict,
            legend=legend,
            **legend_kwargs,
        )


def boxplot(
    ax: plt.Axes,
    data: ad.AnnData | pd.DataFrame,
    grouping_column: list[str] | None = None,
    value_column: list[str] | None = None,
    direct_columns: list[str] | None = None,
    color: tuple = BaseColors.get("blue"),
    color_dict: dict | None = None,
    subgroup_column: str | None = None,
    width: float = 0.4,
    legend: str | mpl.legend.Legend | None = None,
    legend_kwargs: dict | None = None,
) -> None:
    """Plot a box plot from a DataFrame or AnnData object

    Creates a box plot showing the distribution of values for grouped data.
    Each box shows the median, quartiles, and outliers for values within a group.
    Boxes have semi-transparent fill with opaque black outlines, medians, whiskers, and caps.

    Two modes of operation:
    1. **Grouping mode**: Use grouping_column/value_column to group data by categories
    2. **Direct mode**: Use direct_columns to compare multiple columns directly

    Parameters
    ----------
    ax
        Matplotlib axes object to plot on.
    data
        Data containing grouping and value columns or direct columns to plot.
    grouping_column
        Column containing the groups to compare (categorical).
        Used with value_column for grouped comparisons. By default None.
    value_column
        Column whose values should be plotted (numeric).
        Used with grouping_column for grouped comparisons. By default None.
    direct_columns
        List of column names to compare directly. Each column becomes a separate box.
        Overrides grouping_column and value_column. By default None.
    color
        Default color for all boxes. By default BaseColors.get("blue").
    color_dict
        Dictionary mapping group labels to specific colors. Overrides the color
        parameter for specified groups. By default None.

    Returns
    -------
    None

    Examples
    --------
    Grouped comparison (long format):

    .. code-block:: python

        import pandas as pd
        import anndata as ad
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt

        data = pd.DataFrame({"intensity": [1, 2, 3, 4, 5, 6, 7]})
        obs = pd.DataFrame({"group": ["A", "A", "B", "B", "B", "C", "C"]})
        adata = ad.AnnData(X=data.values, obs=obs, var=pd.DataFrame(index=data.columns))

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.boxplot(
            ax=ax,
            data=adata,
            grouping_column="group",
            value_column="intensity",
            color_dict={"A": "red", "B": "green", "C": "blue"},
        )

    Direct column comparison (wide format):

    .. code-block:: python

        import pandas as pd
        import anndata as ad
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt

        data = pd.DataFrame({"protein1": [1, 2, 3], "protein2": [4, 5, 6], "protein3": [7, 8, 9]})
        adata = ad.AnnData(X=data.values, var=pd.DataFrame(index=data.columns))

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.boxplot(
            ax=ax,
            data=adata,
            direct_columns=["protein1", "protein2", "protein3"],
        )

    Notes
    -----
    - Boxes show median (center line), quartiles (box edges), and outliers (points)
    - Whiskers extend to 1.5 * IQR or the most extreme non-outlier point
    - Boxes have 50% transparency with opaque black outlines
    - When using direct_columns, each column's distribution is shown separately
    - Missing values (NaN) are excluded from the distribution calculations
    """
    data, labels, positions, color_keys = _extract_groupwise_plotting_data(
        data=data,
        grouping_column=grouping_column,
        value_column=value_column,
        direct_columns=direct_columns,
        subgroup_column=subgroup_column,
        width=width,
    )

    boxes = ax.boxplot(
        x=data,
        positions=positions,
        widths=width,
        patch_artist=True,
    )

    # Styling of boxes
    for color_key, box in zip(color_keys, boxes["boxes"], strict=False):
        current_color = color_dict.get(color_key, config["na_color"]) if color_dict else color
        box.set_facecolor(mcolors.to_rgba(current_color, alpha=0.5))
        box.set(linewidth=config["linewidths"]["large"])
        box.set_edgecolor(BaseColors.get("black"))

    # Styping of medians
    for _, median in zip(labels, boxes["medians"], strict=False):
        median.set(color=BaseColors.get("black"))
        median.set(linewidth=config["linewidths"]["large"])

    # Styling of whiskers
    for _, whisker in zip(labels * 2, boxes["whiskers"], strict=False):
        whisker.set(color=BaseColors.get("black"))
        whisker.set(linewidth=config["linewidths"]["large"])

    # Styling of caps
    for _, cap in zip(labels * 2, boxes["caps"], strict=False):
        cap.set(color=BaseColors.get("black"))
        cap.set(linewidth=config["linewidths"]["large"])

    # One label per position in case subgroups returned repeated entries
    unique_labels = list(dict.fromkeys(labels))

    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels)

    legend_kwargs = legend_kwargs or {}
    if legend is not None and color_dict is not None:
        add_legend_to_axes(
            ax=ax,
            levels=color_dict,
            legend=legend,
            **legend_kwargs,
        )


def violinplot(
    ax: plt.Axes,
    data: ad.AnnData | pd.DataFrame,
    grouping_column: list[str] | None = None,
    value_column: list[str] | None = None,
    direct_columns: list[str] | None = None,
    color: tuple = BaseColors.get("blue"),
    color_dict: dict | None = None,
    subgroup_column: str | None = None,
    width: float = 0.4,
    legend: str | mpl.legend.Legend | None = None,
    legend_kwargs: dict | None = None,
) -> None:
    """Plot a violin plot from a DataFrame or AnnData object

    Creates a violin plot showing the distribution density of values for grouped data.
    Each violin shows the kernel density estimation of the distribution, along with
    medians, quartiles, and min/max whiskers. Violins have semi-transparent fill with
    opaque black outlines and internal statistical markers.

    Two modes of operation:
    1. **Grouping mode**: Use grouping_column/value_column to group data by categories
    2. **Direct mode**: Use direct_columns to compare multiple columns directly

    Parameters
    ----------
    ax
        Matplotlib axes object to plot on.
    data
        Data containing grouping and value columns or direct columns to plot.
    grouping_column
        Column containing the groups to compare (categorical).
        Used with value_column for grouped comparisons. By default None.
    value_column
        Column whose values should be plotted (numeric).
        Used with grouping_column for grouped comparisons. By default None.
    direct_columns
        List of column names to compare directly. Each column becomes a separate violin.
        Overrides grouping_column and value_column. By default None.
    color
        Default color for all violins. By default BaseColors.get("blue").
    color_dict
        Dictionary mapping group labels to specific colors. Overrides the color
        parameter for specified groups. By default None.

    Returns
    -------
    None

    Examples
    --------
    Grouped comparison (long format):

    .. code-block:: python

        import pandas as pd
        import anndata as ad
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt

        data = pd.DataFrame({"intensity": [1, 2, 3, 4, 5, 6, 7]})
        obs = pd.DataFrame({"group": ["A", "A", "B", "B", "B", "C", "C"]})
        adata = ad.AnnData(X=data.values, obs=obs, var=pd.DataFrame(index=data.columns))

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.violinplot(
            ax=ax,
            data=adata,
            grouping_column="group",
            value_column="intensity",
            color_dict={"A": "red", "B": "green", "C": "blue"},
        )

    Direct column comparison (wide format):

    .. code-block:: python

        import pandas as pd
        import anndata as ad
        from alphapepttools.pl.figure import create_figure
        import alphapepttools as apt

        data = pd.DataFrame({"protein1": [1, 2, 3], "protein2": [4, 5, 6], "protein3": [7, 8, 9]})
        adata = ad.AnnData(X=data.values, var=pd.DataFrame(index=data.columns))

        fig, axm = create_figure(1, 1, figsize=(6, 4))
        ax = axm.next()
        apt.pl.violinplot(
            ax=ax,
            data=adata,
            direct_columns=["protein1", "protein2", "protein3"],
        )

    Notes
    -----
    - Violins show kernel density estimation of the distribution
    - Internal markers show median, quartiles, and min/max values
    - Violins have 50% transparency with opaque black outlines
    - When using direct_columns, each column's distribution is shown separately
    - Missing values (NaN) are excluded from the distribution calculations
    """
    data, labels, positions, color_keys = _extract_groupwise_plotting_data(
        data=data,
        grouping_column=grouping_column,
        value_column=value_column,
        direct_columns=direct_columns,
        subgroup_column=subgroup_column,
        width=width,
    )

    violins = ax.violinplot(
        dataset=data,
        positions=positions,
        widths=width,
        showmedians=True,
    )

    # Styling of violins
    for color_key, violin in zip(color_keys, violins["bodies"], strict=False):
        current_color = color_dict.get(color_key, config["na_color"]) if color_dict else color
        violin.set_facecolor(mcolors.to_rgba(current_color, alpha=0.5))
        violin.set_edgecolor(BaseColors.get("black"))
        violin.set_linewidth(config["linewidths"]["large"])
        violin.set_alpha(None)  # Reset any global alpha

    # Styling of medians
    violins["cmedians"].set(color=BaseColors.get("black"))
    violins["cmedians"].set(linewidth=config["linewidths"]["large"])

    # Styling of min and max whiskers and the central bar
    violins["cmins"].set(color=BaseColors.get("black"))
    violins["cmins"].set(linewidth=config["linewidths"]["large"])
    violins["cmaxes"].set(color=BaseColors.get("black"))
    violins["cmaxes"].set(linewidth=config["linewidths"]["large"])
    violins["cbars"].set(color=BaseColors.get("black"))
    violins["cbars"].set(linewidth=config["linewidths"]["large"])

    # One label per position in case subgroups returned repeated entries
    unique_labels = list(dict.fromkeys(labels))

    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels)

    legend_kwargs = legend_kwargs or {}
    if legend is not None and color_dict is not None:
        add_legend_to_axes(
            ax=ax,
            levels=color_dict,
            legend=legend,
            **legend_kwargs,
        )


def rank_median_plot(
    data: ad.AnnData | pd.DataFrame,
    ax: plt.Axes,
    layer: str = "X",
    color: str = "blue",
    color_map_column: str | None = None,
    color_column: str | None = None,
    palette: list[str | tuple] | None = None,
    color_dict: dict[str, str | tuple] | None = None,
    legend: str | mpl.legend.Legend | None = None,
    scatter_kwargs: dict | None = None,
) -> None:
    """Rank plot showing median intensities across samples.

    Computes the median intensity for each feature (protein/peptide) across all samples,
    ranks them from highest to lowest, and creates a scatter plot with rank on the x-axis
    and median intensity on the y-axis (log-scale). Useful for visualizing the dynamic
    range of detected features and identifying highly abundant vs low-abundance features.

    Parameters
    ----------
    data
        AnnData or DataFrame containing intensity values.
    ax
        Matplotlib axes object to plot on.
    layer
        The AnnData layer to use for calculating median intensities. Default is "X".
    color
        Single color for all points. Overridden by color_map_column or color_column.
    color_map_column
        Column in data.var (for AnnData) to use for color encoding. Values are mapped
        to colors using the palette or color_dict. Overrides the color parameter.
    color_column
        Column in data.var (for AnnData) containing actual color values (hex, RGBA, etc.).
        Overrides both color and color_map_column parameters.
    palette
        List of colors to use for color encoding. If None, a default palette is used.
    color_dict
        Dictionary mapping category values to specific colors. If provided, palette is ignored.
    legend
        Legend specification. Use "auto" to automatically create a legend from color_map_column.
    scatter_kwargs
        Additional keyword arguments passed to matplotlib scatter function (e.g., alpha, s).

    Examples
    --------
    Basic rank plot with single color:

    .. code-block:: python

        fig, ax = plt.subplots()
        apt.pl.rank_median_plot(
            data=adata,
            ax=ax,
            color=BaseColors.get("blue"),
            scatter_kwargs={"alpha": 0.7},
        )

    Color by protein category:

    .. code-block:: python

        fig, ax = plt.subplots()
        apt.pl.rank_median_plot(
            data=adata,
            ax=ax,
            color_map_column="protein_type",
            color_dict={"protein_type_A": "red", "protein_type_B": "green", "protein_type_C": "blue"},
            legend="auto",
            scatter_kwargs={"s": 20},
        )

    Notes
    -----
    - The y-axis is automatically set to log scale
    - Features are ranked from highest to lowest median intensity
    - For AnnData objects, var annotations can be used for coloring via color_map_column
    - This is a convenience wrapper around the scatter() method with automatic data preparation

    """
    scatter_kwargs = scatter_kwargs or {}

    if ax is None:
        _, axm = create_figure()
        ax = axm.next()

    if layer != "X" and layer not in data.layers:
        raise ValueError(f"Layer {layer} not found in AnnData object")

    # Use AnnData's dataframe extraction to get the values + annotations
    values = (data.to_df() if layer == "X" else data.to_df(layer=layer)) if isinstance(data, ad.AnnData) else data

    # compute medians and sort
    medians = values.median(axis=0).sort_values(ascending=False).to_frame(name="median")

    # Retain information about the proteins
    medians = medians.join(data.var) if isinstance(data, ad.AnnData) else medians
    medians["rank"] = np.arange(1, len(medians) + 1)

    # call scatter to create the rank plot
    scatter(
        data=medians,
        x_column="rank",
        y_column="median",
        color=color,
        color_column=color_column,
        color_map_column=color_map_column,
        legend=legend,
        palette=palette,
        color_dict=color_dict,
        ax=ax,
        scatter_kwargs=scatter_kwargs,
    )

    # Adjust scale and labelling
    ax.set_yscale("log")

    label_axes(
        ax,
        xlabel="Rank",
        ylabel="Median Intensity",
    )


def plot_pca(
    data: ad.AnnData,
    x_column: int = 1,
    y_column: int = 2,
    color: str = "blue",
    color_map_column: str | None = None,
    color_column: str | None = None,
    dim_space: str = "obs",
    embeddings_name: str | None = None,
    method: Literal["pca", "bpca"] = "pca",
    label: bool = False,  # noqa: FBT001, FBT002
    label_column: str | None = None,
    ax: plt.Axes | None = None,
    palette: list[str | tuple] | None = None,
    color_dict: dict[str, str | tuple] | None = None,
    legend: str | mpl.legend.Legend | None = None,
    scatter_kwargs: dict | None = None,
) -> None:
    """PCA scatter plot showing principal component projections.

    Visualizes PCA results by plotting two principal components against each other.
    The function retrieves PCA embeddings from the AnnData object based on the dim_space
    parameter: use "obs" for sample projections (most common, shows how samples relate)
    or "var" for feature projections (shows how features/genes relate). Axes are
    automatically labeled with explained variance percentages.

    Parameters
    ----------
    data
        AnnData object containing PCA results (must have run PCA first).
    x_column
        Principal component number for x-axis (1-indexed, so 1 = PC1, 2 = PC2, etc.).
    y_column
        Principal component number for y-axis (1-indexed).
    color
        Single color for all points. Overridden by color_map_column or color_column.
    color_map_column
        Column in data.obs (for dim_space="obs") or data.var (for dim_space="var") to use
        for color encoding. Values are mapped to colors using palette or color_dict.
        Overrides the color parameter.
    color_column
        Column containing actual color values (hex, RGBA, etc.). Overrides both color
        and color_map_column parameters.
    dim_space
        PCA space to visualize:
        - "obs": Sample projections (default) - shows samples in PC space
        - "var": Feature projections - shows features/genes in PC space
    embeddings_name
        Custom embeddings name if non-default name was used in the PCA function.
        If None, uses default naming convention ("X_pca_obs" or "X_pca_var").
    method
        The method used for dimensionality reduction. Options are "pca" or "bpca" with "pca" as the default.
        This is used to construct the default keys if `embeddings_name` is None.
    label
        Whether to add text labels to points in the scatter plot.
    label_column
        Column to use for point labels. If None and label=True, uses the index
        (data.obs.index for dim_space="obs", data.var.index for dim_space="var").
    ax
        Matplotlib axes to plot on. If None, a new figure is created.
    palette
        List of colors for color encoding. If None, uses default qualitative palette.
    color_dict
        Dictionary mapping category values to specific colors. Overrides palette.
    legend
        Legend specification. Use "auto" to create legend from color_map_column.
    scatter_kwargs
        Additional keyword arguments passed to matplotlib scatter (e.g., s, alpha).

    Examples
    --------
    Basic PCA plot with sample coloring:

    .. code-block:: python

        fig, ax = plt.subplots()
        Plots.plot_pca(
            data=adata,
            ax=ax,
            x_column=1,
            y_column=2,
            color_map_column="replicate",
            legend="auto",
        )

    PCA with custom PC axes and labels:

    .. code-block:: python

        fig, ax = plt.subplots()
        Plots.plot_pca(
            data=adata,
            ax=ax,
            x_column=2,  # PC2
            y_column=3,  # PC3
            label=True,
            label_column="sample_id",
            color_map_column="treatment",
            color_dict={"Control": "gray", "Drug": "red"},
        )

    Feature space PCA (var projection):

    .. code-block:: python

        # Show how proteins/genes relate to each other in PC space
        fig, ax = plt.subplots()
        Plots.plot_pca(
            data=adata,
            ax=ax,
            x_column=1,
            y_column=2,
            dim_space="var",  # Feature projection instead of sample
            color_map_column="protein_type",
            scatter_kwargs={"s": 20, "alpha": 0.6},
        )

    Notes
    -----
    - PCA must be run on the AnnData object before calling this function
    - Axis labels automatically include explained variance percentages (e.g., "PC1 (45.2%)")
    - dim_space="obs" retrieves sample projections from obsm (most common usage)
    - dim_space="var" retrieves feature projections from varm (less common)
    - PC numbers are 1-indexed: x_column=1 corresponds to the first principal component
    - This is a convenience wrapper around scatter() with automatic PCA data extraction

    """
    scatter_kwargs = scatter_kwargs or {}

    adata_pca = extract_pca_anndata(
        data,
        dim_space=dim_space,
        embeddings_name=embeddings_name,
        expression_columns=color_map_column,
        method=method,
    )

    # get the explained variance ratio for the dimensions (for axis labels)
    var_dim1 = adata_pca.var["variance_ratio"][f"pc_{x_column}"]
    var_dim1 = round(var_dim1 * 100, 2)
    var_dim2 = adata_pca.var["variance_ratio"][f"pc_{y_column}"]
    var_dim2 = round(var_dim2 * 100, 2)

    # check pc_x and pc_y are valid
    n_pcs = adata_pca.shape[1]
    if x_column < 1 or x_column > n_pcs or y_column < 1 or y_column > n_pcs:
        raise ValueError(f"pc_x and pc_y are out of bounds, must be between 1 and {n_pcs}")

    scatter(
        data=adata_pca,
        x_column=f"pc_{x_column}",
        y_column=f"pc_{y_column}",
        color=color,
        color_column=color_column,
        color_map_column=color_map_column,
        legend=legend,
        palette=palette,
        color_dict=color_dict,
        ax=ax,
        scatter_kwargs=scatter_kwargs,
    )

    # add labels if requested
    if label:
        # For labeling, we need to consider the appropriate observation space
        if dim_space == "obs":
            labels = data.obs.index if label_column is None else data_column_to_array(data, label_column)
        else:  # dim_space == "var"
            labels = data.var.index if label_column is None else data_column_to_array(data, label_column)

        # Create a DataFrame with the PCA coordinates and labels for the new label_plot interface
        label_df = pd.DataFrame({"x": adata_pca.X[:, x_column - 1], "y": adata_pca.X[:, y_column - 1], "label": labels})

        label_plot(
            ax=ax,
            data=label_df,
            x_column="x",
            y_column="y",
            label_column="label",
            x_anchors=None,
        )

    # set axislabels
    label_axes(ax, xlabel=f"PC{x_column} ({var_dim1}%)", ylabel=f"PC{y_column} ({var_dim2}%)")


def scree_plot(
    adata: ad.AnnData | pd.DataFrame,
    ax: plt.Axes,
    n_pcs: int = 20,
    dim_space: str = "obs",
    color: str = "blue",
    embeddings_name: str | None = None,
    method: Literal["pca", "bpca"] = "pca",
    scatter_kwargs: dict | None = None,
) -> None:
    """Scree plot showing explained variance for each principal component.

    Creates a scatter plot displaying the percentage of variance explained by each
    principal component. Useful for determining how many PCs capture most of the
    variation in the data and for deciding how many components to retain for analysis.

    Parameters
    ----------
    adata
        AnnData object containing PCA results (must have run PCA first).
    ax
        Matplotlib axes object to plot on.
    n_pcs
        Number of principal components to plot on the x-axis.
    dim_space
        PCA space to retrieve variance from:
        - "obs": Sample space PCA (default) - variance explained across samples
        - "var": Feature space PCA - variance explained across features
    color
        Color for the scatter points.
    embeddings_name
        Custom embeddings name if non-default name was used in the PCA function.
        If None, uses default naming convention.
    method
        The method used for dimensionality reduction. Options are "pca" or "bpca" with "pca" as the default.
        This is used to construct the default keys if `embeddings_name` is None.
    scatter_kwargs
        Additional keyword arguments passed to matplotlib scatter (e.g., s, alpha).

    Examples
    --------
    Basic scree plot:

    .. code-block:: python

        fig, ax = plt.subplots()
        Plots.scree_plot(adata=adata, ax=ax, n_pcs=50)

    Scree plot with custom styling:

    .. code-block:: python

        fig, ax = plt.subplots()
        Plots.scree_plot(adata=adata, ax=ax, n_pcs=30, color="red", scatter_kwargs={"s": 50, "alpha": 0.8})

    Feature space scree plot:

    .. code-block:: python

        # Show variance explained in feature space PCA
        fig, ax = plt.subplots()
        Plots.scree_plot(adata=adata, ax=ax, n_pcs=20, dim_space="var")

    Notes
    -----
    - PCA must be run on the AnnData object before calling this function
    - Y-axis shows percentage of total variance explained by each PC
    - dim_space="obs" shows variance for sample projections (most common)
    - dim_space="var" shows variance for feature projections
    - This is a convenience wrapper around scatter() with automatic variance data extraction

    """
    scatter_kwargs = scatter_kwargs or {}

    # create the dataframe for plotting, X = pcs, y = explained variance
    values = prepare_scree_data_to_plot(adata, n_pcs, dim_space, embeddings_name, method=method)

    scatter(
        data=values,
        x_column="PC",
        y_column="explained_variance_percent",
        ax=ax,
        scatter_kwargs=scatter_kwargs,
        color=color,
    )

    # set labels
    space_suffix = " (samples)" if dim_space == "obs" else " (features)"
    label_axes(ax, xlabel="PC number", ylabel=f"Explained variance (%){space_suffix}")


def plot_pca_loadings(
    data: ad.AnnData | pd.DataFrame,
    ax: plt.Axes,
    dim_space: str = "obs",
    embeddings_name: str | None = None,
    method: Literal["pca", "bpca"] = "pca",
    dim: int = 1,
    nfeatures: int = 20,
    scatter_kwargs: dict | None = None,
) -> None:
    """1D loadings plot showing top features contributing to a principal component.

    Creates a scatter plot displaying the loadings (weights) of the top contributing
    features for a single principal component. Loadings indicate how much each feature
    (gene/protein) contributes to the PC. The plot shows the top N features ranked
    by absolute loading value.

    Parameters
    ----------
    data
        AnnData object containing PCA results (must have run PCA first).
    ax
        Matplotlib axes object to plot on.
    dim_space
        PCA space to retrieve loadings from:
        - "obs": Sample space PCA (default) - shows which features drive sample separation
        - "var": Feature space PCA - shows which samples drive feature separation
    embeddings_name
        Custom embeddings name if non-default name was used in the PCA function.
        If None, uses default naming convention.
    method
        The method used for dimensionality reduction. Options are "pca" or "bpca" with "pca" as the default.
        This is used to construct the default keys if `embeddings_name` is None.
    dim
        Principal component number to show loadings for (1-indexed, so 1 = PC1, 2 = PC2, etc.).
    nfeatures
        Number of top features (by absolute loading value) to display.
    scatter_kwargs
        Additional keyword arguments passed to matplotlib scatter (e.g., s, alpha).

    Examples
    --------
    Basic loadings plot for PC1:

    .. code-block:: python

        fig, ax = plt.subplots()
        Plots.plot_pca_loadings(
            data=adata,
            ax=ax,
            dim=1,
            nfeatures=20,
        )

    Loadings plot for PC3 with more features:

    .. code-block:: python

        fig, ax = plt.subplots()
        Plots.plot_pca_loadings(data=adata, ax=ax, dim=3, nfeatures=30, scatter_kwargs={"s": 50, "alpha": 0.8})

    Feature space loadings (var projection):

    .. code-block:: python

        # Show which samples most influence feature PC1
        fig, ax = plt.subplots()
        Plots.plot_pca_loadings(
            data=adata,
            ax=ax,
            dim=1,
            dim_space="var",
            nfeatures=15,
        )

    Notes
    -----
    - PCA must be run on the AnnData object before calling this function
    - Features are ranked by absolute loading value (magnitude, not sign)
    - Y-axis shows feature names, X-axis shows loading values
    - dim_space="obs" shows feature loadings (most common - which proteins/genes matter)
    - dim_space="var" shows sample loadings (which samples matter)
    - This is a convenience wrapper around scatter() with automatic loadings data extraction

    """
    scatter_kwargs = scatter_kwargs or {}

    top_loadings = prepare_pca_1d_loadings_data_to_plot(
        data=data,
        dim_space=dim_space,
        embeddings_name=embeddings_name,
        method=method,
        dim=dim,
        nfeatures=nfeatures,
    )

    scatter(
        data=top_loadings,
        x_column="dim_loadings",
        y_column="index_int",
        ax=ax,
        scatter_kwargs=scatter_kwargs,
    )

    # set axis labels
    space_suffix = " features" if dim_space == "obs" else " samples"
    label_axes(ax, xlabel=f"PC{dim} loadings", ylabel=f"Top{space_suffix}")
    ax.set_yticks(top_loadings["index_int"])
    ax.set_yticklabels(top_loadings["feature"], rotation=0, ha="right")


def plot_pca_loadings_2d(
    data: ad.AnnData | pd.DataFrame,
    ax: plt.Axes,
    dim_space: str = "obs",
    embeddings_name: str | None = None,
    method: Literal["pca", "bpca"] = "pca",
    pc_x: int = 1,
    pc_y: int = 2,
    nfeatures: int = 20,
    *,
    add_labels: bool = True,
    add_lines: bool = False,
    scatter_kwargs: dict | None = None,
) -> None:
    """2D loadings plot showing top features contributing to two principal components.

    Creates a scatter plot displaying the first two principal component loadings against each other.
    Loadings indicate how much each feature (gene/protein) contributes to each PC. The plot shows
    all features used in the PCA as grey points, with the top N features (by absolute loading value)
    highlighted in blue. Optionally, labels can be added to the top features.

    Parameters
    ----------
    data
        AnnData to plot.
    ax
        Matplotlib axes object to plot on.
    dim_space
        The dimension space used in PCA. Can be either "obs" (default) for sample projection
        or "var" for feature projection. By default "obs".
    embeddings_name
        The custom embeddings name used in PCA. If None, uses default naming convention. By default None.
    method
        The method used for dimensionality reduction. Options are "pca" or "bpca" with "pca" as the default.
        This is used to construct the default keys if `embeddings_name` is None.
    pc_x
        The PC principal component index to plot on the x axis, by default 1.
        Corresponds to the principal component order, the first principal is 1 (1-indexed,
        i.e. the first PC is 1, not 0).
    pc_y
        The principal component index to plot on the y axis, by default 2.
        Corresponds to the principal component order, the first principal is 1 (1-indexed,
        i.e. the first PC is 1, not 0).
    nfeatures
        The number of top absolute loadings features to label from each component, by default 20
    add_labels
        Whether to add feature labels of the top `nfeatures` loadings. by default `True`.
    add_lines
        If True, draw lines connecting the origin (0,0) to the points representing the top `nfeatures` loadings.
        Default is `False`.
    scatter_kwargs
        Additional keyword arguments for the matplotlib scatter function. By default None.

    Examples
    --------
    Basic 2D PCA loadings plot:

    .. code-block:: python

        fig, ax = plt.supplots()
        Plots.plot_pca_loadings_2d(
            data=adata,
            ax=ax,
            pc_x=1,
            pc_y=2,
            nfeatures=20,
            add_labels=True,
            add_lines=True,
            scatter_kwargs=None,
        )

    Notes
    -----
    - PCA must be run on the AnnData object before calling this function
    - Features are ranked by absolute loading value (magnitude, not sign)
    - X and Y axes show loading values for the specified principal components
    - dim_space="obs" shows feature loadings (most common - which proteins/genes matter)
    - dim_space="var" shows sample loadings (which samples matter)
    - This is a convenience wrapper around scatter() with automatic loadings data extraction

    """
    scatter_kwargs = scatter_kwargs or {}

    # Generate the correct loadings key name

    loadings_df = prepare_pca_2d_loadings_data_to_plot(
        data=data,
        embeddings_name=embeddings_name,
        method=method,
        pc_x=pc_x,
        pc_y=pc_y,
        nfeatures=nfeatures,
        dim_space=dim_space,
    )

    # plot the loadings of all features (used in PCA) first
    scatter_kwargs.update({"alpha": 0.3, "s": 10, "edgecolors": "none"})

    scatter(
        data=loadings_df,
        x_column="dim1_loadings",
        y_column="dim2_loadings",
        ax=ax,
        color="grey",
        scatter_kwargs=scatter_kwargs,
    )

    loadings_top = loadings_df[loadings_df["is_top"]]

    # plot the top features on top
    scatter_kwargs.update({"alpha": 1, "s": 20, "edgecolors": "none"})

    scatter(
        data=loadings_top,
        x_column="dim1_loadings",
        y_column="dim2_loadings",
        ax=ax,
        color="blue",
        scatter_kwargs=scatter_kwargs,
    )

    # add labels to the top features
    if add_labels:
        label_plot(
            ax=ax,
            data=loadings_top,
            x_column="dim1_loadings",
            y_column="dim2_loadings",
            label_column="feature",
            x_anchors=None,
            label_kwargs={"fontsize": config["font_sizes"]["medium"], "ha": "center", "va": "bottom"},
            line_kwargs={"color": BaseColors.get("black"), "linewidth": config["linewidths"]["medium"]},
        )
    # draw lines from the origin to the top features if specified
    if add_lines:
        for xi, yi in zip(loadings_top["dim1_loadings"], loadings_top["dim2_loadings"], strict=False):
            ax.plot([0, xi], [0, yi], color="gray", linestyle="-", linewidth=0.2)

    # set axis labels
    space_suffix = " (samples)" if dim_space == "obs" else " (features)"
    label_axes(ax, xlabel=f"PC{pc_x}{space_suffix}", ylabel=f"PC{pc_y}{space_suffix}")


def volcano(  # noqa: C901
    # Required data parameters
    data: ad.AnnData | pd.DataFrame,
    x_column: str = "log2fc",
    y_column: str = "-log10(p_value)",
    # Core plotting parameters
    ax: plt.Axes | None = None,
    layers: list[tuple] | None = None,
    color_dict: dict[str, str | tuple] | None = None,
    # Volcano-specific thresholds
    x_thresholds: float | tuple = (-1, 1),
    y_thresholds: float | tuple = -np.log10(0.05),
    # Labeling parameters
    label_layers: list[str] | None = None,
    display_id_column: str | None = None,
    max_labels: int | None = None,
    x_label_anchors: list[float] | None = None,
    y_display_start: float = 1,
    y_padding_factor: float = 4,
    # Plot limits
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    # Styling parameters
    scatter_kwargs: dict | None = None,
    line_kwargs: dict | None = None,
    label_kwargs: dict | None = None,
    legend: str | mpl.legend.Legend | None = None,
    legend_kwargs: dict | None = None,
    # Default layer parameters
    default_color: str | tuple = BaseColors.get("grey"),
    default_group: str = "data",
) -> None:
    """Create a volcano plot for differential expression visualization

    Volcano plots visualize differential expression results by plotting fold change (x-axis)
    against statistical significance (y-axis). This function creates layered scatter plots
    with threshold lines and optional point labeling.

    Parameters
    ----------
    data
        Data containing expression values and statistics
    x_column
        Column name for x-axis values (typically log fold change)
    y_column
        Column name for y-axis values (typically -log10 p-value)
    ax
        Axes to plot on. If None, creates new figure
    layers
        List of layer specifications for hierarchical plotting. Each tuple contains
        (column_name, value(s), color_key[, scatter_kwargs]). Points are plotted
        in reverse order (first layer on top). Example:
        [("gene_type", "housekeeping", "hk_color"),
         ("significance", "significant", "sig_color", {"s": 100})]
    color_dict
        Maps color keys from layers to actual colors. Example:
        {"hk_color": "blue", "sig_color": "red"}
    x_thresholds
        X-axis values for vertical threshold lines. Default (-1, 1) for
        fold change cutoffs
    y_thresholds
        Y-axis values for horizontal threshold lines. Default (-log10(0.05),)
        for p-value cutoff
    label_layers
        Color keys of layers to label. Only points in these layers will have
        text labels added
    display_id_column
        Column containing labels to display. If None, uses data index
    max_labels
        Maximum number of labels to show. Labels are prioritized by y-value
    x_label_anchors
        X-positions to anchor labels to (for alignment). If None, labels
        appear at data point positions
    y_display_start
        Starting y-position for stacked labels (1=top, 0=bottom). Default 1
    y_padding_factor
        Vertical spacing multiplier between stacked labels. Default 4
    xlims
        X-axis limits. If None, calculated from data with padding
    ylims
        Y-axis limits. If None, calculated from data with padding
    scatter_kwargs
        Additional arguments passed to scatter plot (e.g., {"s": 50, "alpha": 0.5})
    line_kwargs
        Additional arguments for threshold lines (e.g., {"linewidth": 2, "linestyle": "--"})
    label_kwargs
        Additional arguments for axis labels
    legend
        Legend specification. If "auto", creates legend from color_dict
    legend_kwargs
        Additional arguments for legend
    default_color
        Color for points not matching any layer. Default grey
    default_group
        Name for the default layer containing unassigned points. Default "data"

    Returns
    -------
    None

    See Also
    --------
    layered_plot : Core layering functionality
    add_lines : Add threshold lines
    label_plot : Add text labels

    Notes
    -----
    The layering system ensures each point appears in exactly one layer.
    Points are assigned to the first matching layer in the list.
    Unassigned points go to the default layer (plotted in background).

    Examples
    --------
    Create a volcano plot with differential expression data:

    .. code-block:: python

        import numpy as np
        import pandas as pd
        import alphapepttools as apt
        from alphapepttools.pl import BaseColors

        # Generate example differential expression data
        rng = np.random.default_rng(seed=42)
        testx = rng.normal(0, 1, 300)
        testy = -np.cos(testx) + rng.normal(0, 0.2, 300)
        testp = 10 ** -(testy - min(testy))

        data = pd.DataFrame(
            {
                "id": [f"P{10000 + i}" for i in range(300)],
                "gene": [f"gene_{i}" for i in range(300)],
                "log2fc": testx,
                "pval": testp,
                "neg_log10pval": -np.log10(testp),
            }
        )
        data.index = data["id"]

        # Add differential expression status
        data["diff_exp_status"] = data["log2fc"].apply(
            lambda x: "upregulated" if x > 1 else ("downregulated" if x < -1 else "unchanged")
        )

        # Mark first 10 genes as proteins of interest
        data["label"] = "other"
        data.loc[data.index[:10], "label"] = "POI"

        # Define specific proteins to highlight
        pois = ["P10291", "P10292", "P10293", "P10294", "P10295"]

        # Define visualization layers (plotted in reverse order)
        plot_layers = [
            ("id", pois, "POI_hypothesis"),  # Specific hypothesis proteins on top
            ("label", "POI", "POI"),  # General POI proteins
            ("diff_exp_status", "upregulated", "upregulated"),  # Upregulated
            ("diff_exp_status", "downregulated", "downregulated"),  # Downregulated
            ("diff_exp_status", "unchanged", "unchanged"),  # Background
        ]

        # Define colors for each layer
        color_dict = {
            "upregulated": BaseColors.get("orange"),
            "downregulated": BaseColors.get("blue"),
            "unchanged": BaseColors.get("grey"),
            "POI": "black",
            "POI_hypothesis": BaseColors.get("purple", lighten=0.7),
        }

        # Specify which layers to label
        label_layers = ["POI", "POI_hypothesis"]

        # Create volcano plot
        apt.pl.volcano(
            data=data,
            x_column="log2fc",
            y_column="neg_log10pval",
            color_dict=color_dict,
            layers=plot_layers,
            label_layers=label_layers,
            x_label_anchors=[-3.5, 3.5],  # Anchor labels to left/right
            y_padding_factor=1.7,  # Vertical spacing between labels
            y_display_start=0.75,  # Start labels at 75% from bottom
            xlims=(-6, 6),
        )

    This creates a volcano plot where:

    - Background points (unchanged) appear in grey
    - Differentially expressed genes are colored orange (up) or blue (down)
    - Proteins of interest (POI) are highlighted in black
    - Specific hypothesis proteins are emphasized in purple on top
    - Only POI and hypothesis proteins receive text labels

    """
    data = data.copy()

    label_kwargs = label_kwargs or {}
    scatter_kwargs = scatter_kwargs or {}
    line_kwargs = line_kwargs or {}
    legend_kwargs = legend_kwargs or {}
    x_thresholds = (x_thresholds,) if isinstance(x_thresholds, (int, float)) else x_thresholds
    y_thresholds = (y_thresholds,) if isinstance(y_thresholds, (int, float)) else y_thresholds

    # Generate a plotting config
    plot_config = make_scatter_config(
        data=data,
        x_column=x_column,
        y_column=y_column,
        scatter_kwargs=scatter_kwargs,
    )

    # get limits
    if xlims is None:
        xlims = _get_plot_lims(
            data_column_to_array(data, x_column),
            padding_factor=1.1,
            sym="max",
        )
    if ylims is None:
        ylims = _get_plot_lims(
            data_column_to_array(data, y_column),
            padding_factor=1.1,
            set_left=0,
        )

    # Initialize alphapepttools style figure and axes if not provided
    if ax is None:
        _, axm = create_figure()
        ax = axm.next()

    # Add decoration to volcanoplot prior to setting limits
    add_lines(ax=ax, linetype="vline", intercepts=list(x_thresholds), line_kwargs=line_kwargs)
    add_lines(ax=ax, linetype="hline", intercepts=list(y_thresholds), line_kwargs=line_kwargs)

    # Visualize & keep track of display layers for labelling
    global_layer_indices = layered_plot(
        ax=ax,
        base_config=plot_config,
        layers=layers,
        color_dict=color_dict,
        xlims=xlims,
        ylims=ylims,
        return_glob_layer_indices=True,
        default_color=default_color,
        default_color_key=default_group,
    )

    # Labeling
    if label_layers is not None:
        idxs_to_label = []
        for layer_idxs, _, color_key, _ in global_layer_indices:
            if color_key in label_layers:
                idxs_to_label.extend(layer_idxs)

        if idxs_to_label:
            # Extract points for labeling
            label_df = coerce_to_dataframe(subset_data(data, idxs_to_label))

            # Sort by y values (descending) to minimize crossing lines between labels and points
            label_df = label_df.sort_values(y_column, ascending=False)

            # Get display labels, fall back to index if no display_id_column is provided
            if display_id_column is not None:
                label_df["__label__"] = data_column_to_array(label_df, display_id_column)
            else:
                label_df["__label__"] = data_index_to_array(label_df, "obs")

            # Apply max_labels if specified
            if max_labels is not None and len(label_df) > max_labels:
                label_df = label_df.head(max_labels)

            label_plot(
                ax=ax,
                data=label_df,
                x_column=x_column,
                y_column=y_column,
                label_column="__label__",
                y_display_start=y_display_start,
                x_anchors=x_label_anchors,
                y_padding_factor=y_padding_factor,
            )

    # Sensible default axis labeling
    label_axes(
        ax=ax,
        xlabel=x_column,
        ylabel=y_column,
        title="Volcano Plot",
        **label_kwargs,
    )

    # Add legend from color dict
    if legend is not None and color_dict is not None:
        add_legend_to_axes(
            ax=ax,
            levels=color_dict,
            **legend_kwargs,
        )
