# plots.py

# Main plotting submodule with a private method for generating legends, and a
# Plots class containing class methods to generate plots. The proposed
# layout for plotting functions is such that they accept AnnData objects and dataframes.
# When columns to plot are specified for an AnnData object, the _adata_column_to_array()
# function first tries to find the column in the var_names (i.e. the columns of the actual
# data), and then in the obs.columns (for example, when plotting a numeric value from X and
# coloring it by a metadata column from obs, see 03_basic_workflow.ipynb).

import logging
from collections import Counter
from collections.abc import Callable
from typing import Any

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
from alphapepttools.pp.data import data_column_to_array, data_columns_to_df
from alphapepttools.tl.plot_data_handling import (
    extract_pca_anndata,
    prepare_pca_1d_loadings_data_to_plot,
    prepare_pca_2d_loadings_data_to_plot,
    prepare_scree_data_to_plot,
)

# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = defaults.plot_settings.to_dict()


def _extract_groupwise_plotting_data(
    data: ad.AnnData | pd.DataFrame,
    grouping_column: str | None = None,
    value_column: str | None = None,
    direct_columns: list[str] | None = None,
) -> tuple[list[list], list[str], list[int]]:
    """Extract data for group-wise plotting (violin, bar, box plots)

    Transforms long-format data into the list-of-lists format required by
    matplotlib's violin, bar, and box plot functions. Each sublist contains
    the values for one group.

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

    Returns
    -------
    tuple[list[list], list[str], list[int]]
        (data_lists, labels, positions) for plotting

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
    """
    if direct_columns is not None:
        if grouping_column is not None or value_column is not None:
            logger.info("'direct_columns' provided, ignoring 'grouping_column' and 'value_column' parameters.")
        df = data_columns_to_df(data, columns=direct_columns)[direct_columns]  # ensure order
        df = df.melt(var_name="variable", value_name="value")
        grouping_column, value_column = "variable", "value"
    else:
        df = data_columns_to_df(data, columns=[grouping_column, value_column])

    # Determine groups
    groups_to_plot = df[grouping_column].dropna().unique().tolist()

    # Extract data for each group
    data_lists = []
    labels = []
    positions = []

    for i, group in enumerate(groups_to_plot):
        group_data = df[df[grouping_column] == group][value_column].dropna()
        if not group_data.empty:
            data_lists.append(group_data.tolist())
            labels.append(group)
            positions.append(i + 1)

    return data_lists, labels, positions


def add_lines(
    ax: plt.Axes,
    intercepts: float | list[float | int],
    linetype: str = "vline",
    color: str = "black",
    linestyle: str = "--",
    linewidth: float | None = None,
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
        Line width, defaults to `config["linewidths"]["medium"]`
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
    line_kwargs = line_kwargs or {}
    color = line_kwargs.pop("color", color)
    linestyle = line_kwargs.pop("linestyle", linestyle)
    if linewidth is None:
        linewidth = config["linewidths"]["medium"]
    linewidth = line_kwargs.pop("linewidth", linewidth)

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
        palette = ["#blue", "#lightblue", "#red"]
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


def _drop_nans_from_plot_arrays(
    x_values: np.ndarray,
    y_values: np.ndarray,
    labels: np.ndarray | list[str],
) -> tuple:
    """Remove entries where either x or y is NaN, applying same mask to labels

    Creates a mask from x and y arrays where neither value is NaN, then
    applies this same mask to filter all three arrays consistently.

    Parameters
    ----------
    x_values
        X coordinates for plotting
    y_values
        Y coordinates for plotting
    labels
        Labels corresponding to each x,y pair

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (filtered_x, filtered_y, filtered_labels) with NaN entries removed

    Example
    -------
    .. code-block:: python

        x = np.array([1, 2, np.nan, 4])
        y = np.array([5, np.nan, 7, 8])
        labels = np.array(["A", "B", "C", "D"])

        x_clean, y_clean, labels_clean = _drop_nans_from_plot_arrays(x, y, labels)
        # Returns: ([1, 4], [5, 8], ["A", "D"])
        # Drops index 1 (y is NaN) and index 2 (x is NaN)
    """
    # Missing x or y values are breaking and should be dropped
    keep_mask = ~np.logical_or(pd.isna(x_values), pd.isna(y_values))

    return x_values[keep_mask], y_values[keep_mask], labels[keep_mask]


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
    or anchored to specific x-positions. Lines are drawn from the data points to the labels.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to add the labels to.
    data : pd.DataFrame | ad.AnnData
        Data containing the x, y, and label columns.
    x_column : str
        Column name for x-coordinates.
    y_column : str
        Column name for y-coordinates.
    label_column : str
        Column name for labels.
    x_anchors : list[int | float] | np.ndarray | None, optional
        x-coordinates of the anchors to use for the labels. If None, labels are placed at the x-coordinates of the data points. By default None.
    label_kwargs : dict | None, optional
        Additional keyword arguments for the label text, by default None.
    line_kwargs : dict | None, optional
        Additional keyword arguments for the line connecting the label to the data point, by default None.
    label_parser : Callable | None, optional
        Function to parse the labels, by default None. This is useful to convert
        labels from a computation-context to presentation context, e.g. a column
        like upregulated_proteins could be shown as "Upregulated Proteins" in the plot.
    y_display_start : float, optional
        Starting point for the y-coordinates of the labels, by default 1. This is used to determine the spacing between labels.
        The y-coordinates of the labels are adjusted to be evenly spaced between the min and max y-coordinates at that anchor.
        This is useful for avoiding label overlap.
    y_padding_factor: float, optional
        Factor to increase or decrease how far apart labels are spread in the y-direction when stacked into a column over x-anchors

    """
    label_kwargs = {"fontsize": config["font_sizes"]["medium"], **(label_kwargs or {})}
    line_kwargs = {"color": BaseColors.get("black"), "linewidth": config["linewidths"]["medium"], **(line_kwargs or {})}
    label_parser = label_parser or (lambda x: x)

    # Extract all needed columns into a DataFrame
    df = data_columns_to_df(data, columns=[x_column, y_column, label_column])

    # Sort by y values (highest to lowest)
    df = df.sort_values(by=y_column, ascending=False)

    # Extract arrays from sorted DataFrame
    x_values = df[x_column].to_numpy()
    y_values = df[y_column].to_numpy()
    labels = df[label_column].to_numpy()

    # Remove any nans
    x_values, y_values, labels = _drop_nans_from_plot_arrays(x_values, y_values, labels)

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


class Plots:
    """Class for creating figures with matplotlib

    Configuration for matplotlib plots is loaded from the defaults module
    as a dictionary and used to generate consistent plots.

    Overview
    --------
    The Plots class provides alphapepttools styled visualization methods
    for proteomics and other biological data. All methods accept either
    pandas DataFrames or AnnData objects and use column names to specify
    data to plot.

    Available Plot Types
    --------------------
    **Distribution plots:**
        - :meth:`histogram`: Histograms with optional color grouping
        - :meth:`violinplot`: Violin plots showing distribution density
        - :meth:`boxplot`: Box plots showing quartiles and outliers
        - :meth:`barplot`: Bar plots with error bars (mean ± std)

    **Relationship plots:**
        - :meth:`scatter`: Scatter plots with flexible coloring options
        - :meth:`rank_median_plot`: Ranked median intensity plots

    **Convenience wrapper plots:**
    These plots summarize common visualization tasks in proteomics for ease of use.
        - :meth:`plot_pca`: PCA scatter plots with optional labeling
        - :meth:`scree_plot`: Eigenvalue/variance explained plots
        - :meth:`plot_pca_loadings`: 1D loading plots for a single PC
        - :meth:`plot_pca_loadings_2d`: 2D loading plots for two PCs

    Common Parameters
    -----------------
    Most plotting methods share these common parameters:

    data : pd.DataFrame or ad.AnnData
        Input data object
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (created in alphapepttools style if not provided)

    Examples
    --------
    Basic scatter plot:

    >>> import pandas as pd
    >>> from alphapepttools.pl import Plots
    >>>
    >>> df = pd.DataFrame(
    ...     {
    ...         "log2fc": [1.5, -2.0, 0.5, 3.0],
    ...         "pvalue": [0.01, 0.001, 0.5, 0.005],
    ...         "significant": ["yes", "yes", "no", "yes"],
    ...     }
    ... )
    >>>
    >>> fig, ax = plt.subplots()
    >>> Plots.scatter(
    ...     data=df,
    ...     x_column="log2fc",
    ...     y_column="pvalue",
    ...     color_map_column="significant",
    ...     color_dict={"yes": "red", "no": "gray"},
    ...     ax=ax,
    ... )

    Distribution comparison with violin plot:

    >>> # Compare protein intensities across conditions
    >>> Plots.violinplot(
    ...     ax=ax,
    ...     data=adata,
    ...     grouping_column="condition",
    ...     value_column="intensity",
    ...     color_dict={"Control": "blue", "Treatment": "red"},
    ... )

    PCA with sample labels:

    >>> # PCA plot with sample type coloring and labels
    >>> Plots.plot_pca(
    ...     data=adata,
    ...     x_column=1,  # PC1
    ...     y_column=2,  # PC2
    ...     color_map_column="sample_type",
    ...     label=True,
    ...     label_column="sample_id",
    ...     ax=ax,
    ... )

    Notes
    -----
    - All methods are class methods and can be called directly without instantiation
    - Color handling is flexible: direct colors, categorical mapping, or continuous gradients
    - Plots automatically handle both DataFrame and AnnData inputs
    - Configuration is loaded as a dictionary from defaults.plot_settings via 'defaults.plot_settings.to_dict()'

    See Also
    --------
    :func:`add_legend_to_axes` : Add legends to plots
    :func:`label_plot` : Add labels to scatter plots
    :func:`add_lines` : Add reference lines to plots
    """

    def __init__(
        self,
        config: dict = defaults.plot_settings.to_dict(),
    ):
        self.config = config

    @classmethod
    def histogram(
        cls,
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
        data : pd.DataFrame | ad.AnnData
            Data to plot, must contain the value_column and optionally
            the color_map_column for grouping.
        value_column : str
            Column containing numeric values to plot in the histogram.
        color_map_column : str, optional
            Column for categorical grouping. Each unique value gets its own
            colored histogram overlay. NaN values are converted to strings.
        bins : int, optional
            Number of bins for the histogram. Default is 10.
        ax : plt.Axes, optional
            Matplotlib axes to plot on. If None, a new figure is created.
        color : str, optional
            Single color for ungrouped histogram. Default is "blue".
        palette : list[tuple], optional
            Color palette for grouped histograms. Defaults to qualitative palette.
        color_dict : dict[str, str | tuple], optional
            Explicit mapping of groups to colors. Overrides palette if provided.
        legend : str | mpl.legend.Legend, optional
            If "auto", creates legend for grouped data. Can also pass existing Legend.
        hist_kwargs : dict, optional
            Additional arguments for matplotlib.hist() like:
            - alpha: transparency (0-1)
            - histtype: 'bar', 'step', 'stepfilled'
            - edgecolor: outline color
            - linewidth: outline width
        legend_kwargs : dict, optional
            Additional arguments for legend like title, loc, fontsize.
        xlim : tuple[float, float], optional
            X-axis limits as (min, max).
        ylim : tuple[float, float], optional
            Y-axis limits as (min, max).

        Returns
        -------
        None

        Examples
        --------
        Simple histogram:

        >>> Plots.histogram(data=df, value_column="intensity", bins=30, color="skyblue", ax=ax)

        Grouped histogram with transparency:

        >>> Plots.histogram(
        ...     data=df,
        ...     value_column="values",
        ...     color_map_column="condition",
        ...     bins=20,
        ...     legend="auto",
        ...     hist_kwargs={"alpha": 0.7, "histtype": "stepfilled"},
        ...     legend_kwargs={"title": "Condition"},
        ...     ax=ax,
        ... )

        Custom color mapping:

        >>> Plots.histogram(
        ...     data=example_df,
        ...     value_column="values",
        ...     color_map_column="levels",
        ...     color_dict={"A": "red", "B": "blue", "C": "green"},
        ...     bins=20,
        ...     ax=ax,
        ...     legend="auto",
        ...     hist_kwargs={"alpha": 0.7, "histtype": "stepfilled", "edgecolor": "k"},
        ...     legend_kwargs={"title": "Levels", "loc": "upper left"},
        ... )

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

    @classmethod
    def scatter(
        cls,
        data: pd.DataFrame | ad.AnnData,
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
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
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
        data : pd.DataFrame | ad.AnnData
            Data to plot, must contain the x_column and y_column and optionally the color_column or color_map_column.
        x_column : str
            Column in data to plot on the x-axis. Must contain numeric data.
        y_column : str
            Column in data to plot on the y-axis. Must contain numeric data.
        color : str, optional
            Color to use for the scatterplot. By default "blue".
        color_map_column : str, optional
            Column in data to use for color encoding. These values are mapped to the palette or the color_dict (see below).
            Its values cannot contain NaNs, therefore color_map_column is coerced to string and missing values replaced by
            a default filler string. Overrides color parameter. By default None.
        color_column : str, optional
            Column in data to plot the colors. This must contain actual color values (RGBA, hex, etc.).
            Overrides color and color_map_column parameters. By default None.
        ax : plt.Axes, optional
            Matplotlib axes object to plot on, if None a new figure is created. By default None.
        palette : list[str | tuple] | matplotlib.colors.Colormap, optional
            List of colors to use for color encoding, if None a default palette is used.
            Can be a matplotlib Colormap for continuous gradients. By default None.
        color_dict : dict[str, str | tuple], optional
            Supercedes palette, a dictionary mapping levels to colors. By default None. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function (s, alpha, edgecolors, etc.). By default None.
        legend_kwargs : dict, optional
            Additional keyword arguments for the matplotlib legend function. By default None.
        xlim : tuple[float, float], optional
            Limits for the x-axis. By default None.
        ylim : tuple[float, float], optional
            Limits for the y-axis. By default None.

        Returns
        -------
        None

        Examples
        --------
        Simple scatter with single color:

        >>> Plots.scatter(data=df, x_column="log2fc", y_column="pvalue", color="red", ax=ax)

        Categorical coloring with automatic palette:

        >>> Plots.scatter(
        ...     data=adata,
        ...     x_column="values",
        ...     y_column="values2",
        ...     color_map_column="levels3",
        ...     legend="auto",
        ...     ax=ax,
        ...     palette=None,  # Uses default qualitative palette
        ... )

        Many categories with repeating palette:

        >>> # Default palette repeats for many distinct values
        >>> Plots.scatter(
        ...     data=df,
        ...     x_column="values",
        ...     y_column="values2",
        ...     color_map_column="levels2",  # e.g., 50 distinct values
        ...     ax=ax,
        ... )

        Sequential colormap for unique colors:

        >>> # Avoid repetition with sequential colormap
        >>> Plots.scatter(
        ...     data=adata,
        ...     x_column="values",
        ...     y_column="values2",
        ...     color_map_column="levels2",  # Many distinct values
        ...     ax=ax,
        ...     palette=colors.BaseColormaps.get("sequential"),
        ... )

        Quantitative gradient with numeric data:

        >>> # Numeric values mapped to continuous gradient
        >>> Plots.scatter(
        ...     data=adata,
        ...     x_column="values",
        ...     y_column="values2",
        ...     color_map_column="levels3",  # Numeric column
        ...     legend="auto",
        ...     ax=ax,
        ...     palette=colors.BaseColormaps.get("sequential"),
        ... )

        Custom color dictionary:

        >>> Plots.scatter(
        ...     data=df,
        ...     x_column="x",
        ...     y_column="y",
        ...     color_map_column="significance",
        ...     color_dict={"significant": "red", "not_sig": "grey", "borderline": "orange"},
        ...     legend="auto",
        ...     scatter_kwargs={"s": 50, "alpha": 0.7},
        ...     ax=ax,
        ... )

        Direct color values from column:

        >>> # color_column contains actual color values like "#FF0000" or "red"
        >>> Plots.scatter(
        ...     data=df,
        ...     x_column="x",
        ...     y_column="y",
        ...     color_column="my_colors",  # Contains hex/RGB/color names
        ...     ax=ax,
        ... )

        Notes
        -----
        - Points are ordered by color frequency (most frequent in back) for better visibility
        - Unmapped values in color_dict default to grey
        - NaN values in color columns are handled as strings
        """
        scatter_kwargs = scatter_kwargs or {}
        legend_kwargs = legend_kwargs or {}
        DEFAULT_GROUP = "data"
        DEFAULT_COLOR = BaseColors.get("blue")

        if ax is None:
            _, axm = create_figure()
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
                    or get_color_mapping(
                        values=color_map_column_array, palette=palette or BasePalettes.get("qualitative")
                    )
                )

                for level in set(color_map_column_array) - set(color_dict):
                    color_dict[level] = BaseColors.get("grey")

                color_values = np.array([color_dict[level] for level in color_map_column_array], dtype=object)
        else:
            color_dict = {DEFAULT_GROUP: color or DEFAULT_COLOR}
            color_values = np.array([color_dict[DEFAULT_GROUP]] * len(data))

        # Handle ordering of plotting arrays by string: order by the frequency of the color column
        counts = Counter([str(cv) for cv in color_values])
        order = np.argsort([counts[str(cv)] for cv in color_values])[::-1]
        x_values = data_column_to_array(data, x_column)[order]
        y_values = data_column_to_array(data, y_column)[order]
        color_values = np.array(color_values)[order]

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

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    @classmethod
    def barplot(
        cls,
        ax: plt.Axes,
        data: ad.AnnData | pd.DataFrame,
        grouping_column: list[str] | None = None,
        value_column: list[str] | None = None,
        direct_columns: list[str] | None = None,
        color: tuple = BaseColors.get("blue"),
        color_dict: dict | None = None,
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
        ax : plt.Axes
            Matplotlib axes object to plot on.
        data : ad.AnnData | pd.DataFrame
            Data containing grouping and value columns or direct columns to plot.
        grouping_column : str | None, optional
            Column containing the groups to compare (categorical).
            Used with value_column for grouped comparisons. By default None.
        value_column : str | None, optional
            Column whose values should be plotted (numeric).
            Used with grouping_column for grouped comparisons. By default None.
        direct_columns : list[str] | None, optional
            List of column names to compare directly. Each column becomes a separate bar.
            Overrides grouping_column and value_column. By default None.
        color : tuple, optional
            Default color for all bars. By default BaseColors.get("blue").
        color_dict : dict | None, optional
            Dictionary mapping group labels to specific colors. Overrides the color
            parameter for specified groups. By default None.

        Returns
        -------
        None

        Examples
        --------
        Grouped comparison (categories from one column):

        >>> # Compare values across different treatment groups (long table format)
        >>> Plots.barplot(
        ...     ax=ax,
        ...     data=adata,
        ...     grouping_column="treatment",  # Categories: "Control", "Drug_A", "Drug_B"
        ...     value_column="expression",  # Numeric values to compare
        ...     color_dict={"Control": "gray", "Drug_A": "blue", "Drug_B": "red"},
        ... )

        Direct column comparison:

        >>> # Compare multiple measurement columns directly (wide table format)
        >>> Plots.barplot(
        ...     ax=ax,
        ...     data=adata,
        ...     direct_columns=["protein1", "protein2", "protein3"],  # Each column becomes a bar
        ...     color=BaseColors.get("green"),
        ... )

        Notes
        -----
        - Error bars show standard deviation of values within each group
        - Bars have 50% transparency with opaque black outlines
        - When using direct_columns, each column's mean is calculated across all rows
        - Missing values (NaN) are excluded from mean and std calculations
        """
        data, labels, positions = _extract_groupwise_plotting_data(
            data=data,
            grouping_column=grouping_column,
            value_column=value_column,
            direct_columns=direct_columns,
        )

        means = [pd.Series(d).mean() for d in data]
        stds = [pd.Series(d).std() for d in data]

        bars = ax.bar(
            x=positions,
            height=means,
            yerr=stds,
            capsize=5,
            align="center",
            width=0.5,
        )

        # Styling of bars
        for label, bar in zip(labels, bars, strict=False):
            current_color = color_dict.get(label, config["na_color"]) if color_dict else color
            bar.set_facecolor(mcolors.to_rgba(current_color, alpha=0.5))
            bar.set_edgecolor(BaseColors.get("black"))
            bar.set(linewidth=config["linewidths"]["large"])

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

    @classmethod
    def boxplot(
        cls,
        ax: plt.Axes,
        data: ad.AnnData | pd.DataFrame,
        grouping_column: list[str] | None = None,
        value_column: list[str] | None = None,
        direct_columns: list[str] | None = None,
        color: tuple = BaseColors.get("blue"),
        color_dict: dict | None = None,
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
        ax : plt.Axes
            Matplotlib axes object to plot on.
        data : ad.AnnData | pd.DataFrame
            Data containing grouping and value columns or direct columns to plot.
        grouping_column : str | None, optional
            Column containing the groups to compare (categorical).
            Used with value_column for grouped comparisons. By default None.
        value_column : str | None, optional
            Column whose values should be plotted (numeric).
            Used with grouping_column for grouped comparisons. By default None.
        direct_columns : list[str] | None, optional
            List of column names to compare directly. Each column becomes a separate box.
            Overrides grouping_column and value_column. By default None.
        color : tuple, optional
            Default color for all boxes. By default BaseColors.get("blue").
        color_dict : dict | None, optional
            Dictionary mapping group labels to specific colors. Overrides the color
            parameter for specified groups. By default None.

        Returns
        -------
        None

        Examples
        --------
        Grouped comparison (categories from one column):

        >>> # Compare values across different treatment groups (long table format)
        >>> Plots.boxplot(
        ...     ax=ax,
        ...     data=adata,
        ...     grouping_column="treatment",  # Categories: "Control", "Drug_A", "Drug_B"
        ...     value_column="expression",  # Numeric values to compare
        ...     color_dict={"Control": "gray", "Drug_A": "blue", "Drug_B": "red"},
        ... )

        Direct column comparison:

        >>> # Compare multiple measurement columns directly (wide table format)
        >>> Plots.boxplot(
        ...     ax=ax,
        ...     data=adata,
        ...     direct_columns=["protein1", "protein2", "protein3"],  # Each column becomes a box
        ...     color=BaseColors.get("green"),
        ... )

        Notes
        -----
        - Boxes show median (center line), quartiles (box edges), and outliers (points)
        - Whiskers extend to 1.5 * IQR or the most extreme non-outlier point
        - Boxes have 50% transparency with opaque black outlines
        - When using direct_columns, each column's distribution is shown separately
        - Missing values (NaN) are excluded from the distribution calculations
        """
        data, labels, positions = _extract_groupwise_plotting_data(
            data=data,
            grouping_column=grouping_column,
            value_column=value_column,
            direct_columns=direct_columns,
        )

        boxes = ax.boxplot(
            x=data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
        )

        # Styling of boxes
        for label, box in zip(labels, boxes["boxes"], strict=False):
            current_color = color_dict.get(label, config["na_color"]) if color_dict else color
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

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

    @classmethod
    def violinplot(
        cls,
        ax: plt.Axes,
        data: ad.AnnData | pd.DataFrame,
        grouping_column: list[str] | None = None,
        value_column: list[str] | None = None,
        direct_columns: list[str] | None = None,
        color: tuple = BaseColors.get("blue"),
        color_dict: dict | None = None,
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
        ax : plt.Axes
            Matplotlib axes object to plot on.
        data : ad.AnnData | pd.DataFrame
            Data containing grouping and value columns or direct columns to plot.
        grouping_column : str | None, optional
            Column containing the groups to compare (categorical).
            Used with value_column for grouped comparisons. By default None.
        value_column : str | None, optional
            Column whose values should be plotted (numeric).
            Used with grouping_column for grouped comparisons. By default None.
        direct_columns : list[str] | None, optional
            List of column names to compare directly. Each column becomes a separate violin.
            Overrides grouping_column and value_column. By default None.
        color : tuple, optional
            Default color for all violins. By default BaseColors.get("blue").
        color_dict : dict | None, optional
            Dictionary mapping group labels to specific colors. Overrides the color
            parameter for specified groups. By default None.

        Returns
        -------
        None

        Examples
        --------
        Grouped comparison (categories from one column):

        >>> # Compare values across different treatment groups (long table format)
        >>> Plots.violinplot(
        ...     ax=ax,
        ...     data=adata,
        ...     grouping_column="treatment",  # Categories: "Control", "Drug_A", "Drug_B"
        ...     value_column="expression",  # Numeric values to compare
        ...     color_dict={"Control": "gray", "Drug_A": "blue", "Drug_B": "red"},
        ... )

        Direct column comparison:

        >>> # Compare multiple measurement columns directly (wide table format)
        >>> Plots.violinplot(
        ...     ax=ax,
        ...     data=adata,
        ...     direct_columns=["protein1", "protein2", "protein3"],  # Each column becomes a violin
        ...     color=BaseColors.get("purple"),
        ... )

        Notes
        -----
        - Violins show kernel density estimation of the distribution
        - Internal markers show median, quartiles, and min/max values
        - Violins have 50% transparency with opaque black outlines
        - When using direct_columns, each column's distribution is shown separately
        - Missing values (NaN) are excluded from the distribution calculations
        """
        data, labels, positions = _extract_groupwise_plotting_data(
            data=data,
            grouping_column=grouping_column,
            value_column=value_column,
            direct_columns=direct_columns,
        )

        violins = ax.violinplot(
            dataset=data,
            positions=positions,
            widths=0.5,
            showmedians=True,
        )

        # Styling of violins
        for label, violin in zip(labels, violins["bodies"], strict=False):
            current_color = color_dict.get(label, config["na_color"]) if color_dict else color
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

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

    @classmethod
    def rank_median_plot(
        cls,
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
        data : ad.AnnData | pd.DataFrame
            AnnData or DataFrame containing intensity values.
        ax : plt.Axes
            Matplotlib axes object to plot on.
        layer : str, default="X"
            The AnnData layer to use for calculating median intensities. Default is "X".
        color : str, default="blue"
            Single color for all points. Overridden by color_map_column or color_column.
        color_map_column : str, optional
            Column in data.var (for AnnData) to use for color encoding. Values are mapped
            to colors using the palette or color_dict. Overrides the color parameter.
        color_column : str, optional
            Column in data.var (for AnnData) containing actual color values (hex, RGBA, etc.).
            Overrides both color and color_map_column parameters.
        palette : list[str | tuple], optional
            List of colors to use for color encoding. If None, a default palette is used.
        color_dict : dict[str, str | tuple], optional
            Dictionary mapping category values to specific colors. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend specification. Use "auto" to automatically create a legend from color_map_column.
        scatter_kwargs : dict, optional
            Additional keyword arguments passed to matplotlib scatter function (e.g., alpha, s).

        Examples
        --------
        Basic rank plot with single color:

        .. code-block:: python

            fig, ax = plt.subplots()
            Plots.rank_median_plot(
                data=adata,
                ax=ax,
                color=BaseColors.get("blue"),
                scatter_kwargs={"alpha": 0.7},
            )

        Color by protein category:

        .. code-block:: python

            fig, ax = plt.subplots()
            Plots.rank_median_plot(
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

        # call the Plots.scatter method to create the rank plot
        cls.scatter(
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

    @classmethod
    def plot_pca(
        cls,
        data: ad.AnnData,
        x_column: int = 1,
        y_column: int = 2,
        color: str = "blue",
        color_map_column: str | None = None,
        color_column: str | None = None,
        dim_space: str = "obs",
        embeddings_name: str | None = None,
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
        data : ad.AnnData
            AnnData object containing PCA results (must have run PCA first).
        x_column : int, default=1
            Principal component number for x-axis (1-indexed, so 1 = PC1, 2 = PC2, etc.).
        y_column : int, default=2
            Principal component number for y-axis (1-indexed).
        color : str, default="blue"
            Single color for all points. Overridden by color_map_column or color_column.
        color_map_column : str, optional
            Column in data.obs (for dim_space="obs") or data.var (for dim_space="var") to use
            for color encoding. Values are mapped to colors using palette or color_dict.
            Overrides the color parameter.
        color_column : str, optional
            Column containing actual color values (hex, RGBA, etc.). Overrides both color
            and color_map_column parameters.
        dim_space : str, default="obs"
            PCA space to visualize:
            - "obs": Sample projections (default) - shows samples in PC space
            - "var": Feature projections - shows features/genes in PC space
        embeddings_name : str, optional
            Custom embeddings name if non-default name was used in the PCA function.
            If None, uses default naming convention ("X_pca_obs" or "X_pca_var").
        label : bool, default=False
            Whether to add text labels to points in the scatter plot.
        label_column : str, optional
            Column to use for point labels. If None and label=True, uses the index
            (data.obs.index for dim_space="obs", data.var.index for dim_space="var").
        ax : plt.Axes, optional
            Matplotlib axes to plot on. If None, a new figure is created.
        palette : list[str | tuple], optional
            List of colors for color encoding. If None, uses default qualitative palette.
        color_dict : dict[str, str | tuple], optional
            Dictionary mapping category values to specific colors. Overrides palette.
        legend : str | mpl.legend.Legend, optional
            Legend specification. Use "auto" to create legend from color_map_column.
        scatter_kwargs : dict, optional
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
            data, dim_space=dim_space, embeddings_name=embeddings_name, expression_columns=color_map_column
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

        cls.scatter(
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
            # Prepare data for labeling based on dim_space
            if dim_space == "obs":
                if label_column is None:
                    # Use index as labels
                    adata_pca.obs["_label"] = data.obs.index
                    label_col = "_label"
                else:
                    # Copy the label column to adata_pca
                    adata_pca.obs[label_column] = data_column_to_array(data, label_column)
                    label_col = label_column
            elif label_column is None:
                # Use index as labels
                adata_pca.obs["_label"] = data.var.index
                label_col = "_label"
            else:
                # Copy the label column to adata_pca
                adata_pca.obs[label_column] = data_column_to_array(data, label_column)
                label_col = label_column

            label_plot(
                ax=ax,
                data=adata_pca,
                x_column=f"pc_{x_column}",
                y_column=f"pc_{y_column}",
                label_column=label_col,
                x_anchors=None,
            )

        # set axislabels
        label_axes(ax, xlabel=f"PC{x_column} ({var_dim1}%)", ylabel=f"PC{y_column} ({var_dim2}%)")

    @classmethod
    def scree_plot(
        cls,
        adata: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        n_pcs: int = 20,
        dim_space: str = "obs",
        color: str = "blue",
        embeddings_name: str | None = None,
        scatter_kwargs: dict | None = None,
    ) -> None:
        """Scree plot showing explained variance for each principal component.

        Creates a scatter plot displaying the percentage of variance explained by each
        principal component. Useful for determining how many PCs capture most of the
        variation in the data and for deciding how many components to retain for analysis.

        Parameters
        ----------
        adata : ad.AnnData | pd.DataFrame
            AnnData object containing PCA results (must have run PCA first).
        ax : plt.Axes
            Matplotlib axes object to plot on.
        n_pcs : int, default=20
            Number of principal components to plot on the x-axis.
        dim_space : str, default="obs"
            PCA space to retrieve variance from:
            - "obs": Sample space PCA (default) - variance explained across samples
            - "var": Feature space PCA - variance explained across features
        color : str, default="blue"
            Color for the scatter points.
        embeddings_name : str, optional
            Custom embeddings name if non-default name was used in the PCA function.
            If None, uses default naming convention.
        scatter_kwargs : dict, optional
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
        values = prepare_scree_data_to_plot(adata, n_pcs, dim_space, embeddings_name)

        cls.scatter(
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

    @classmethod
    def plot_pca_loadings(
        cls,
        data: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        dim_space: str = "obs",
        embeddings_name: str | None = None,
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
        data : ad.AnnData | pd.DataFrame
            AnnData object containing PCA results (must have run PCA first).
        ax : plt.Axes
            Matplotlib axes object to plot on.
        dim_space : str, default="obs"
            PCA space to retrieve loadings from:
            - "obs": Sample space PCA (default) - shows which features drive sample separation
            - "var": Feature space PCA - shows which samples drive feature separation
        embeddings_name : str, optional
            Custom embeddings name if non-default name was used in the PCA function.
            If None, uses default naming convention.
        dim : int, default=1
            Principal component number to show loadings for (1-indexed, so 1 = PC1, 2 = PC2, etc.).
        nfeatures : int, default=20
            Number of top features (by absolute loading value) to display.
        scatter_kwargs : dict, optional
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
            dim=dim,
            nfeatures=nfeatures,
        )

        cls.scatter(
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

    @classmethod
    def plot_pca_loadings_2d(
        cls,
        data: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        dim_space: str = "obs",
        embeddings_name: str | None = None,
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
        data : ad.AnnData
            AnnData to plot.
        ax : plt.Axes
            Matplotlib axes object to plot on.
        dim_space : str, optional
            The dimension space used in PCA. Can be either "obs" (default) for sample projection or "var" for feature projection. By default "obs".
        embeddings_name : str | None, optional
            The custom embeddings name used in PCA. If None, uses default naming convention. By default None.
        pc_x : int
            The PC principal component index to plot on the x axis, by default 1. Corresponds to the principal component order, the first principal is 1 (1-indexed, i.e. the first PC is 1, not 0).
        pc_y : int
            The principal component index to plot on the y axis, by default 2. Corresponds to the principal component order, the first principal is 1 (1-indexed, i.e. the first PC is 1, not 0).
        nfeatures : int
            The number of top absolute loadings features to label from each component, by default 20
        add_labels : bool
            Whether to add feature labels of the top `nfeatures` loadings. by default `True`.
        add_lines : bool
            If True, draw lines connecting the origin (0,0) to the points representing the top `nfeatures` loadings. Default is `False`.
        scatter_kwargs : dict, optional
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
        loadings_key = f"PCs_{dim_space}" if embeddings_name is None else embeddings_name

        loadings_df = prepare_pca_2d_loadings_data_to_plot(
            data=data, loadings_name=loadings_key, pc_x=pc_x, pc_y=pc_y, nfeatures=nfeatures, dim_space=dim_space
        )

        # plot the loadings of all features (used in PCA) first
        scatter_kwargs.update({"alpha": 0.3, "s": 10, "edgecolors": "none"})

        cls.scatter(
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

        cls.scatter(
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
