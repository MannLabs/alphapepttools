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

from alphatools.pl import defaults
from alphatools.pl.colors import BaseColors, BasePalettes, _get_colors_from_cmap, get_color_mapping
from alphatools.pl.figure import create_figure, label_axes
from alphatools.pl.plot_data_handling import (
    prepare_pca_1d_loadings_data_to_plot,
    prepare_pca_2d_loadings_data_to_plot,
    prepare_pca_data_to_plot,
    prepare_scree_data_to_plot,
)
from alphatools.pp.data import data_column_to_array

# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = defaults.plot_settings.to_dict()


def _extract_columns_to_df(
    data: ad.AnnData | pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Extract selected columns from AnnData or DataFrame.

    This function serves as an adapter upstream of matplotlib plotting functions,
    which frequently accept an array of values. Extracts the requested columns
    from an AnnData object's X and/or obs object & validates there are no duplicates.

    Parameters
    ----------
    data : ad.AnnData | pd.DataFrame
        Input data object.
    columns : list[str] | None, optional
        List of column names to extract. If None, uses all columns (DataFrame)
        or all columns in X (AnnData). Default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the selected columns.

    """
    if isinstance(data, pd.DataFrame):
        columns = columns or data.columns.tolist()
        try:
            dataset = data[columns]
        except KeyError as e:
            raise KeyError(f"Columns {columns} not found in dataframe.") from e

    elif isinstance(data, ad.AnnData):
        if columns is None:
            dataset = data.to_df()
        else:
            # Partition columns by source
            x_cols = [col for col in columns if col in data.var_names]
            obs_cols = [col for col in columns if col in data.obs.columns]

            # Check for duplicate columns across sources
            duplicates = set(x_cols) & set(obs_cols)
            if duplicates:
                raise KeyError(
                    f"Columns {duplicates} found in both AnnData X and obs. Please ensure unique column names."
                )

            # Check for missing columns
            missing_cols = set(columns) - set(x_cols) - set(obs_cols)
            if missing_cols:
                raise KeyError(f"Columns {missing_cols} not found in AnnData X or obs.")

            # Build dataset from available sources
            parts = []
            if x_cols:
                parts.append(data.to_df()[x_cols])
            if obs_cols:
                parts.append(data.obs[obs_cols])

            dataset = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]

    else:
        raise TypeError(f"Expected pd.DataFrame or ad.AnnData, got {type(data)}")

    return dataset


def _extract_groupwise_plotting_data(
    data: ad.AnnData | pd.DataFrame,
    grouping_column: str | None = None,
    value_column: str | None = None,
    direct_columns: list[str] | None = None,
) -> tuple[list[list], list[str], list[int]]:
    """Extract data for group-wise plotting (violin, bar, box plots).

    Transforms long-format data into the list-of-lists format required by
    matplotlib's violin, bar, and box plot functions. Each sublist contains
    the values for one group. Using direct_columns makes each of its columns
    directly correspond to a group.

    Parameters
    ----------
    data : ad.AnnData | pd.DataFrame
        Data containing grouping and value columns
    grouping_column : str
        Column containing the groups to compare
    value_column : str
        Column whose values should be plotted
    direct_columns: list[str] | None
        Overrides grouping_column and value_column: This argument allows for extraction of
        actual columns directly into data_lists, labels and positions.

    Returns
    -------
    tuple[list[list], list[str], list[int]]
        Tuple of (data_lists, labels, positions) for plotting

    Examples
    --------
    >>> import pandas as pd
    >>> from alphatools.pl import _extract_groupwise_plotting_data
    >>> df = pd.DataFrame({
    ...     'group': ['A', 'A', 'B', 'B', 'C'],
    ...     'X1': [1, 2, 3, 4, 5]
    ...     'X2': [5, 4, 3, 2, 1]
    ...     'X3': [1, 2, 3, 4, 5]
    ... })

    >>> # Use grouping column
    >>> data_lists, labels, positions = _extract_groupwise_plotting_data(df, "group", "X1")
    >>> print(data_lists)  # [[1, 2], [3, 4], [5]]
    >>> print(labels)  # ['A', 'B', 'C']
    >>> print(positions)  # [1, 2, 3]

    >>> # Use columns directly
    >>> data_lists, labels, positions = _extract_groupwise_plotting_data(
    ...     df, "group", "X1", direct_columns=["X1", "X2", "X3"]
    ... )
    >>> print(data_lists)  # [[1, 2, 5], [3, 4, 3], [5, 1, 5]]
    >>> print(labels)  # ['X1', 'X2', 'X3']
    >>> print(positions)  # [1, 2, 3]

    """
    if direct_columns is not None:
        if grouping_column is not None or value_column is not None:
            logger.info("'direct_columns' provided, ignoring 'grouping_column' and 'value_column' parameters.")
        df = _extract_columns_to_df(data, columns=direct_columns)[direct_columns]  # ensure order
        df = df.melt(var_name="variable", value_name="value")
        grouping_column, value_column = "variable", "value"
    else:
        df = _extract_columns_to_df(data, columns=[grouping_column, value_column])

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
    linewidth: float = 1,
    line_kwargs: dict | None = None,
) -> None:
    """Add a vertical or horizontal line to a matplotlib axes object

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to add the line to.
    linetype : str
        Type of line to add, either 'vline' or 'hline'.
    intercepts : float | list[float | int]
        Intercepts of the line(s) to add.
    color : str, optional
        Color of the line(s), by default "black".
    linestyle : str, optional
        Linestyle of the line(s), by default "--".
    linewidth : float, optional
        Linewidth of the line(s), by default 1.
    line_kwargs : dict, optional
        Additional keyword arguments for the line function, by default None. Will be overridden by color, linestyle, and linewidth arguments.

    Returns
    -------
    None

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
    """Create legend patches for a matplotlib legend from a value-to-color mapping

    This is a helper function for the add_legend function.
    Matplotlib legends display labelled patches with a defined color. This function
    takes a dictionary of values and colors and returns a list of named patches.

    Parameters
    ----------
    color_dict : dict[str, str | tuple]
        Dictionary of values and colors.

    Returns
    -------
    list[mpl.patches.Patch]
        List of named patches.
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
    """Make a legend and directly add it to a matplotlib axes object.

    Expects a list of named patches.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to add the legend to.
    patches : list[mpl.patches.Patch]
        List of patches to use for the legend.

    Returns
    -------
    None

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
    """Add a legend to an axis object.

    Handle legend creation in three ways:
    1.: 'levels' is a dictionary of levels and colors, in which case these levels and colors are used directly.
    2.: 'levels' is a list of levels, in which case a color palette is used to assign colors to levels. A custom
    palette can be provided, otherwise a default palette is used.
    3.: 'legend' is a matplotlib legend object, which overrides all other options and is added directly to the axes.
    This defaults to 'auto', which directs to the first two cases.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to add the legend to.
    levels : list[str] | dict[str, str | tuple] | None
        List of levels to use for the legend. Duplicates are removed. Colors from the palette are assigned to unique values from this list,
        but no particular color-binding is enforced. If this is a dictionary, the legend contains exactly the labels (keys) and colors (values) provided.
    legend : str | mpl.legend.Legend | None
        Legend to add to the plot. If "auto", a legend is created based on levels. If a Legend object, it is added directly to the axes. By default "auto".
    palette : list[str | tuple] | None
        List of colors to use for the legend. If None, a default palette will be used. By default None. Only relevant when levels is a list, i.e. when matching
        of values to colors happens automatically.
    legend_kwargs : dict, optional
        Additional keyword arguments for the legend, by default {}. This can include 'fontsize', 'title', etc. These kwargs are not enforced if a matplotlib legend object
        is passed as the `legend` parameter.

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
    # Missing x or y values are breaking and should be dropped
    keep_mask = ~np.logical_or(pd.isna(x_values), pd.isna(y_values))

    return x_values[keep_mask], y_values[keep_mask], labels[keep_mask]


def _assign_nearest_anchor_position_to_values(
    values: np.ndarray,
    anchors: list[int | float] | np.ndarray | None,
) -> np.ndarray:
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
    x_values: list | np.ndarray | pd.Series,
    y_values: list | np.ndarray | pd.Series,
    labels: list[str] | np.ndarray | pd.Series,
    x_anchors: list[int | float] | np.ndarray | None = None,
    label_kwargs: dict | None = None,
    line_kwargs: dict | None = None,
    label_parser: Callable | None = None,
    y_display_start: float = 1,
    y_padding_factor: float = 3,
) -> None:
    """Add labels to a 2D axes object

    Add labels to a plot based on x and y coordinates. The labels are either placed near the datapoint
    using the automatic dodging function from adjust_text or anchored to the left or right of the plot,
    where labels below the splitpoint are anchored to the left and labels above the splitpoint are anchored
    to the right.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to add the labels to.
    x_values : list | np.ndarray
        x-coordinates of the labels.
    y_values : list | np.ndarray
        y-coordinates of the labels.
    labels : list[str] | np.ndarray
        Labels to add to the plot.
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

    if not len(x_values) == len(y_values) == len(labels):
        raise ValueError("x_values, y_values, and labels must have the same length")

    # Force the order of labels from highest to lowest
    y_value_order = np.argsort(np.array(y_values))[::-1]
    y_values = np.array(y_values)[y_value_order]
    x_values = np.array(x_values)[y_value_order]
    labels = np.array(labels)[y_value_order]

    # convert to numpy arrays for consistency & remove any nans
    x_values, y_values, labels = _drop_nans_from_plot_arrays(np.array(x_values), np.array(y_values), np.array(labels))

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

    Basic configuration for matplotlib plots is loaded from a YAML file
    and set to generate consistent plots.

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

        Parameters
        ----------
        data : pd.DataFrame | ad.AnnData
            Data to plot, must contain the value_column and optionally the color_column.
        value_column : str
            Column in data to plot as histogram. Must contain numeric data.
        color_map_column : str, optional
            Column in data to use for color encoding. These values are mapped to the palette or the color_dict (see below). Its values cannot contain NaNs, therefore color_map_column is coerced to string and missing values replaced by a default filler string. Overrides color parameter. By default None.
        bins : int, optional
            Number of bins to use for the histogram. By default 10.
        color : str, optional
            Color to use for the histogram. By default "blue".
        ax : plt.Axes, optional
            Matplotlib axes object to plot on, if None a new figure is created. By default None.
        palette : list[tuple], optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
        color_dict: dict[str, str | tuple], optional
            Supercedes palette, a dictionary mapping levels to colors. By default None. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
        hist_kwargs : dict, optional
            Additional keyword arguments for the matplotlib hist function. By default None.
        legend_kwargs : dict, optional
            Additional keyword arguments for the matplotlib legend function. By default None.
        xlim : tuple[float, float], optional
            Limits for the x-axis. By default None.
        ylim : tuple[float, float], optional
            Limits for the y-axis. By default None.

        Returns
        -------
        None

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
            - If palette is a matplotlib colormap: Numerically map values to colors using the colormap. This means that e.g. 1 and 3 will be closer in color than 1 and 10.
            - If palette is not a matplotlib colormap: Treat numeric values as categorical and color as described above.

        - Examples:
            - color_column="my_colors": Points colored by values in "my_colors" column (must contain valid colors)
            - color_map_column="cell_type": Categorical mapping of cell types to colors
            - color_map_column="expression", palette=plt.cm.viridis: Continuous gradient based on expression values


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
            Column in data to use for color encoding. These values are mapped to the palette or the color_dict (see below). Its values cannot contain NaNs, therefore color_map_column is coerced to string and missing values replaced by a default filler string. Overrides color parameter. By default None.
        color_column : str, optional
            Column in data to plot the colors. This must contain actual color values (RGBA, hex, etc.). Overrides color and color_map_column parameters. By default None.
        ax : plt.Axes, optional
            Matplotlib axes object to plot on, if None a new figure is created. By default None.
        palette : list[str | tuple] | matplotlib.colors.Colormap, optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
        color_dict: dict[str, str | tuple], optional
            Supercedes palette, a dictionary mapping levels to colors. By default None. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.
        legend_kwargs : dict, optional
            Additional keyword arguments for the matplotlib legend function. By default None.
        xlim : tuple[float, float], optional
            Limits for the x-axis. By default None.
        ylim : tuple[float, float], optional
            Limits for the y-axis. By default None.

        Returns
        -------
        None

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

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object to plot on.
        data : ad.AnnData | pd.DataFrame
            Data containing grouping and value columns or direct columns to plot.
        grouping_column : list[str] | None, optional
            Column containing the groups to compare. By default None.
        value_column : list[str] | None, optional
            Column whose values should be plotted. By default None.
        direct_columns : list[str] | None, optional
            Overrides grouping_column and value_column. Each column becomes a separate
            bar group. By default None.
        color : tuple, optional
            Default color for all bars. By default BaseColors.get("blue").
        color_dict : dict | None, optional
            Dictionary mapping group labels to specific colors. Overrides the color
            parameter for specified groups. By default None.

        Returns
        -------
        None

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

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object to plot on.
        data : ad.AnnData | pd.DataFrame
            Data containing grouping and value columns or direct columns to plot.
        grouping_column : list[str] | None, optional
            Column containing the groups to compare. By default None.
        value_column : list[str] | None, optional
            Column whose values should be plotted. By default None.
        direct_columns : list[str] | None, optional
            Overrides grouping_column and value_column. Each column becomes a separate
            box plot. By default None.
        color : tuple, optional
            Default color for all boxes. By default BaseColors.get("blue").
        color_dict : dict | None, optional
            Dictionary mapping group labels to specific colors. Overrides the color
            parameter for specified groups. By default None.

        Returns
        -------
        None

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

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object to plot on.
        data : ad.AnnData | pd.DataFrame
            Data containing grouping and value columns or direct columns to plot.
        grouping_column : list[str] | None, optional
            Column containing the groups to compare. By default None.
        value_column : list[str] | None, optional
            Column whose values should be plotted. By default None.
        direct_columns : list[str] | None, optional
            Overrides grouping_column and value_column. Each column becomes a separate
            violin plot. By default None.
        color : tuple, optional
            Default color for all violins. By default BaseColors.get("blue").
        color_dict : dict | None, optional
            Dictionary mapping group labels to specific colors. Overrides the color
            parameter for specified groups. By default None.

        Returns
        -------
        None

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
        """Plot the ranked protein median intensities across all samples using the scatter method

        Parameters
        ----------
        data : ad.AnnData
            AnnData to plot.
        ax : plt.Axes
            Matplotlib axes object to plot on, add labels and logscale the y-axis.
        layer : str
            The AnnData layer to calculate the median value (intensities) across sample. Default is "X"
        color : str, optional
            Color to use for the scatterplot. By default "blue".
        color_map_column : str, optional
            Column in data to use for color encoding. These values are mapped to the palette or the color_dict (see below). Its values cannot contain NaNs, therefore color_map_column is coerced to string and missing values replaced by a default filler string. Overrides color parameter. By default None.
        color_column : str, optional
            Column in data to plot the colors. This must contain actual color values (RGBA, hex, etc.). Overrides color and color_map_column parameters. By default None.
        palette : list[str | tuple], optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
        color_dict: dict[str, str | tuple], optional
            A dictionary mapping levels to colors. By default None. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

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
        ax: plt.Axes,
        pc_x: int = 1,
        pc_y: int = 2,
        dim_space: str = "obs",
        embbedings_name: str | None = None,
        label: bool = False,  # noqa: FBT001, FBT002
        label_column: str | None = None,
        color: str = "blue",
        color_map_column: str | None = None,
        color_column: str | None = None,
        palette: list[str | tuple] | None = None,
        color_dict: dict[str, str | tuple] | None = None,
        legend: str | mpl.legend.Legend | None = None,
        scatter_kwargs: dict | None = None,
    ) -> None:
        """Plot the PCs of a PCA analysis using the scatter method

        Parameters
        ----------
        adata : ad.AnnData
            AnnData to plot.
        ax : plt.Axes
            Matplotlib axes object to plot on.
        pc_x : int
            The PC principal component index to plot on the x axis, by default 1. Corresponds to the principal component order, the first principal is 1 (1-indexed, i.e. the first PC is 1, not 0).
        pc_y : int
            The principal component index to plot on the y axis, by default 2. Corresponds to the principal component order, the first principal is 1 (1-indexed, i.e. the first PC is 1, not 0).
        dim_space : str, optional
            The dimension space used in PCA. Can be either "obs" (default) for sample projection or "var" for feature projection. By default "obs".
        embbedings_name : str | None, optional
            The custom embeddings name used in PCA (given as input for `pca` function in `embbedings_name` ). If None, uses default naming convention. By default None.
        label: bool,
            Whether to add labels to the points in the scatter plot. by default False.
        label_column: str | None = None,
            Column in data.obs to use for labeling the points. If None, and label is True, data.obs.index labels are added. By default None.
        color : str, optional
            Color to use for the scatterplot. By default "blue".
        color_map_column : str, optional
            Column in data to use for color encoding. These values are mapped to the palette or the color_dict (see below). Its values cannot contain NaNs, therefore color_map_column is coerced to string and missing values replaced by a default filler string. Overrides color parameter. By default None.
        color_column : str, optional
            Column in data to plot the colors. This must contain actual color values (RGBA, hex, etc.). Overrides color and color_map_column parameters. By default None.
        palette : list[str | tuple], optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
        color_dict: dict[str, str | tuple], optional
            Supercedes palette, a dictionary mapping levels to colors. By default None. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        pca_coor_df = prepare_pca_data_to_plot(
            data, pc_x, pc_y, dim_space, embbedings_name, color_map_column, label_column, label=label
        )

        # Check if the variance layer exists in uns
        variance_key = f"variance_pca_{dim_space}" if embbedings_name is None else embbedings_name

        if variance_key not in data.uns:
            raise ValueError(
                f"PCA metadata layer '{variance_key}' not found in AnnData object. "
                f"Found layers: {list(data.uns.keys())}"
            )

        # get the explained variance ratio for the dimensions (for axis labels)
        var_dim1 = data.uns[variance_key]["variance_ratio"][pc_x - 1]
        var_dim1 = round(var_dim1 * 100, 2)
        var_dim2 = data.uns[variance_key]["variance_ratio"][pc_y - 1]
        var_dim2 = round(var_dim2 * 100, 2)

        # add color column
        if color_map_column is not None:
            color_values = data_column_to_array(data, color_map_column)
            pca_coor_df[color_map_column] = color_values

        cls.scatter(
            data=pca_coor_df,
            x_column="dim1",
            y_column="dim2",
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

            label_plot(ax=ax, x_values=pca_coor_df["dim1"], y_values=pca_coor_df["dim2"], labels=labels, x_anchors=None)

        # set axislabels
        label_axes(ax, xlabel=f"PC{pc_x} ({var_dim1}%)", ylabel=f"PC{pc_y} ({var_dim2}%)")

    @classmethod
    def scree_plot(
        cls,
        adata: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        n_pcs: int = 20,
        dim_space: str = "obs",
        color: str = "blue",
        embbedings_name: str | None = None,
        scatter_kwargs: dict | None = None,
    ) -> None:
        """Plot the eigenvalues of each of the PCs using the scatter method

        Parameters
        ----------
        adata : ad.AnnData
            AnnData to plot.
        ax : plt.Axes
            Matplotlib axes object to plot on.
        n_pcs : int,
            number of PCs to plot, by default 20
        dim_space : str, optional
            The dimension space used in PCA. Can be either "obs" (default) for sample projection or "var" for feature projection. By default "obs".
        color : str, optional
            Color to use for the scatterplot. By default "blue".
        embbedings_name : str | None, optional
            The custom embeddings name used in PCA. If None, uses default naming convention. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        # create the dataframe for plotting, X = pcs, y = explained variance
        values = prepare_scree_data_to_plot(adata, n_pcs, dim_space, embbedings_name)

        cls.scatter(
            data=values,
            x_column="PC",
            y_column="explained_variance",
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
        embbedings_name: str | None = None,
        dim: int = 1,
        nfeatures: int = 20,
        scatter_kwargs: dict | None = None,
    ) -> None:
        """Plot the gene loadings of a PC using the scatter method

        Parameters
        ----------
        data : ad.AnnData
            AnnData to plot.
        ax : plt.Axes
            Matplotlib axes object to plot on.
        dim_space : str, optional
            The dimension space used in PCA. Can be either "obs" (default) for sample projection or "var" for feature projection. By default "obs".
        embbedings_name : str | None, optional
            The custom embeddings name used in PCA. If None, uses default naming convention. By default None.
        dim : int
            The PC number from which to get loadings, by default 1 (1-indexed, i.e. the first PC is 1, not 0).
        nfeatures : int
            The number of top absolute loadings features to plot, by default 20
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        top_loadings = prepare_pca_1d_loadings_data_to_plot(
            data=data,
            dim_space=dim_space,
            embbedings_name=embbedings_name,
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
        embbedings_name: str | None = None,
        pc_x: int = 1,
        pc_y: int = 2,
        nfeatures: int = 20,
        *,
        add_labels: bool = True,
        add_lines: bool = False,
        scatter_kwargs: dict | None = None,
    ) -> None:
        """Plot the gene loadings of a PC using the scatter method

        Parameters
        ----------
        data : ad.AnnData
            AnnData to plot.
        ax : plt.Axes
            Matplotlib axes object to plot on.
        dim_space : str, optional
            The dimension space used in PCA. Can be either "obs" (default) for sample projection or "var" for feature projection. By default "obs".
        embbedings_name : str | None, optional
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

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        # Generate the correct loadings key name
        loadings_key = f"PCs_{dim_space}" if embbedings_name is None else embbedings_name

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
                x_values=loadings_top["dim1_loadings"],
                y_values=loadings_top["dim2_loadings"],
                labels=loadings_top["feature"],
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
