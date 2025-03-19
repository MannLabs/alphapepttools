# at_plots.py

# Main plotting submodule with a private method for generating legends, and a
# Plots class containing class methods to generate plots. The proposed
# layout for plotting functions is such that they accept AnnData objects and dataframes.
# When columns to plot are specified for an AnnData object, the _adata_column_to_array()
# function first tries to find the column in the var_names (i.e. the columns of the actual
# data), and then in the obs.columns (for example, when plotting a numeric value from X and
# coloring it by a metadata column from obs, see 03_basic_workflow.ipynb).

import logging
from collections.abc import Callable

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pandas.api.types import is_numeric_dtype

from alphatools.pl import defaults
from alphatools.pl.colors import BaseColors, BasePalettes
from alphatools.pl.figure import create_figure
from alphatools.pp.data import _adata_column_to_array

# logging configuration
logging.basicConfig(level=logging.INFO)

config = defaults.plot_settings.to_dict()


def _order_rarest_to_bottom(
    data: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Reorder a DataFrame by the frequency of a column to avoid overplotting rarer classes with more common ones"""
    if column not in data.columns:
        logging.warning(f"Column {column} not found in data. Skipping reordering.")
        return data

    data = data.copy()

    # Order not only by value counts but also lexically to avoid ambivalent ties
    value_counts = data[column].value_counts().sort_index()
    order = value_counts.index

    data[column] = pd.Categorical(data[column], categories=order, ordered=True)

    return data.sort_values(column, ascending=False)


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

    Returns
    -------
    None

    """
    if linetype not in ["vline", "hline"]:
        raise ValueError("linetype must be 'vline' or 'hline'")

    line_kwargs = line_kwargs or {}

    # handle clashes between keyword arguments and line_kwargs
    if "color" in line_kwargs:
        color = line_kwargs.pop("color")
    if "linestyle" in line_kwargs:
        linestyle = line_kwargs.pop("linestyle")
    if "linewidth" in line_kwargs:
        linewidth = line_kwargs.pop("linewidth")

    # handle intercepts and vertical/horizontal lines
    if isinstance(intercepts, float | int):
        intercepts = [intercepts]
    elif isinstance(intercepts, list):
        pass
    else:
        raise TypeError("intercept must be a float or a list of floats")

    line_func = ax.axvline if linetype == "vline" else ax.axhline

    # add lines to ax
    for intercept in intercepts:
        line_func(
            intercept,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            **line_kwargs,
        )


def _make_legend_patches(
    colors: list[str],
    levels: list[str],
) -> list[mpl.patches.Patch]:
    """Create legend patches for a matplotlib legend"""
    patches = []
    for color, level in zip(colors, levels, strict=False):
        patches.append(
            Patch(
                facecolor=color,
                label=level,
                edgecolor=BaseColors.get("grey"),
                linewidth=config["linewidths"]["small"],
            )
        )

    return patches


def make_legend(
    ax: plt.Axes,
    patches: list[mpl.patches.Patch],
    **kwargs,
) -> None:
    """Add a legend to a matplotlib axes object

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

    # handle the title separately
    _legend.set_title(_legend.get_title().get_text(), prop={"size": config["legend"]["title_size"]})


def add_legend(
    ax: plt.Axes,
    levels: list[str] | None,
    colors: list[str] | None,
    legend: str | mpl.legend.Legend | None,
    **legend_kwargs,
) -> None:
    """Parse the legend parameter of a plot method. Either create a legend or try to add the provided one"""
    if isinstance(legend, mpl.legend.Legend):
        try:
            ax.add_artist(legend)
        except Exception:
            logging.exception("Error adding legend. Ignoring legend.")
    elif isinstance(legend, str) and legend == "auto":
        if levels is None or len(levels) == 0:
            logging.warning("No levels provided for legend. Ignoring legend.")
            return

        unique_levels = np.unique(levels)
        colors = colors or BasePalettes.get("qualitative", n=len(unique_levels))
        patches = _make_legend_patches(colors, unique_levels)
        make_legend(ax, patches, **legend_kwargs)
    else:
        logging.warning("Invalid legend parameter. Ignoring legend.")


def label_plot(
    ax: plt.Axes,
    x_values: list | np.ndarray,
    y_values: list | np.ndarray,
    labels: list[str] | np.ndarray,
    x_anchors: list[int | float] | np.ndarray | None = None,
    label_kwargs: dict | None = None,
    line_kwargs: dict | None = None,
    label_parser: Callable | None = None,
    y_display_start: float = 1,
) -> None:
    """Add labels to a 2D axes object

    Add labels to a plot based on x and y coordinates. The labels are either placed near the datapoint
    using the automatic dodging function from adjust_text or anchored to the left or right of the plot,
    where labels below the splitpoint are anchored to the left and labels above the splitpoint are anchored
    to the right.

    """
    label_kwargs = {**(label_kwargs or {}), "fontsize": config["font_sizes"]["medium"]}
    line_kwargs = {**(line_kwargs or {}), "color": BaseColors.get("black"), "linewidth": config["linewidths"]["medium"]}
    label_parser = label_parser or (lambda x: x)

    if not len(x_values) == len(y_values) == len(labels):
        raise ValueError("x_values, y_values, and labels must have the same length")

    # determine label positions based on optional x_anchors
    if x_anchors is not None:
        # x-values are binned to the anchor positions
        label_x_values = []
        for x in x_values:
            anchor_diffs = [abs(anchor - x) for anchor in x_anchors]
            label_x_values.append(x_anchors[np.argmin(anchor_diffs)])

        # TODO: Clean and refactor this block
        # y-values should be distributed evenly between the min and max y-values at that anchor
        fontsize_display = config["font_sizes"]["medium"]
        label_spacing_display = fontsize_display * 1.5
        transform = ax.transData.inverted()
        _, y_spacing_data = transform.transform((0, label_spacing_display)) - transform.transform((0, 0))

        # get a consistent starting point for y values
        bbox = ax.get_window_extent()
        _, upper_bound_data = transform.transform((0, bbox.height * y_display_start))

        label_y_values = []
        for anchor in np.unique(label_x_values):
            current_label_y_values = np.sort(y_values[np.array(label_x_values) == anchor])
            label_y_values.extend([upper_bound_data - y_spacing_data * i for i in range(len(current_label_y_values))])
    else:
        label_x_values = x_values
        label_y_values = y_values

    # generate lines from data values to label positions
    lines = []
    for label, x, y, label_x, label_y in zip(labels, x_values, y_values, label_x_values, label_y_values, strict=False):
        lines.append(((x, label_x), (y, label_y), label))

    # Sort lines by sorting in decreasing label_y order
    lines = sorted(lines, key=lambda line: line[1][1], reverse=True)

    for line in lines:
        ax.plot(line[0], line[1], **line_kwargs)
        alignment = "right" if line[0][0] > line[0][1] else "left"
        label_kwargs["ha"] = alignment
        ax.text(line[0][1], line[1][1], label_parser(line[2]), **label_kwargs)


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
        color_column: str | None = None,
        bins: int = 10,
        ax: plt.Axes | None = None,
        color: str = "blue",
        palette: list[tuple] | None = None,
        legend: str | mpl.legend.Legend | None = None,
        hist_kwargs: dict | None = None,
        legend_kwargs: dict | None = None,
    ) -> None:
        """Plot a histogram from a DataFrame or AnnData object

        Parameters
        ----------
        data : pd.DataFrame | ad.AnnData
            Data to plot, must contain the value_column and optionally the color_column.
        value_column : str
            Column in data to plot as histogram. Must contain numeric data.
        color_column : str, optional
            Column in data to use for color encoding. Overrides color parameter. By default None.
        bins : int, optional
            Number of bins to use for the histogram. By default 10.
        color : str, optional
            Color to use for the histogram. By default "blue".
        ax : plt.Axes, optional
            Matplotlib axes object to plot on, if None a new figure is created. By default None.
        palette : list[tuple], optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
        hist_kwargs : dict, optional
            Additional keyword arguments for the matplotlib hist function. By default None.
        legend_kwargs : dict, optional
            Additional keyword arguments for the matplotlib legend function. By default None.

        Returns
        -------
        None

        """
        hist_kwargs = hist_kwargs or {}
        legend_kwargs = legend_kwargs or {}

        if not ax:
            _, ax = create_figure(1, 1)

        values = _adata_column_to_array(data, value_column)
        if not is_numeric_dtype(values):
            raise ValueError("Value column must contain numeric data")

        if color_column is None:
            color = BaseColors.get(color)
            ax.hist(values, bins=bins, color=color, **hist_kwargs)
        else:
            colors = _adata_column_to_array(data, color_column)

            levels = np.unique(colors)

            if palette is None:
                palette = BasePalettes.get("qualitative", n=len(levels))

            for _color, level in zip(palette, levels, strict=False):
                ax.hist(values[colors == level], bins=bins, color=_color, **hist_kwargs)

            add_legend(ax, legend, palette, levels, **legend_kwargs)

    @classmethod
    def scatter(
        cls,
        data: pd.DataFrame | ad.AnnData,
        y_column: str,
        x_column: str,
        color_column: str | None = None,
        ax: plt.Axes | None = None,
        scatter_kwargs: dict | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """Plot a scatterplot from a DataFrame or AnnData object

        Parameters
        ----------
        data : pd.DataFrame | ad.AnnData
            Data to plot, must contain the x_column and y_column and optionally the color_column.
        x_column : str
            Column in data to plot on the x-axis. Must contain numeric data.
        y_column : str
            Column in data to plot on the y-axis. Must contain numeric data.
        color_column : str, optional
            Column in data to use for color encoding. By default None.
        ax : plt.Axes, optional
            Matplotlib axes object to plot on, if None a new figure is created. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.
        xlim : tuple[float, float], optional
            Limits for the x-axis. By default None.
        ylim : tuple[float, float], optional
            Limits for the y-axis. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        if not ax:
            _, ax = create_figure()

        if color_column and color_column in data.columns:
            data = _order_rarest_to_bottom(data.copy(), color_column)
            colors = _adata_column_to_array(data, color_column)
        else:
            colors = [BaseColors.get("blue")] * len(data)

        x_values = _adata_column_to_array(data, x_column)
        y_values = _adata_column_to_array(data, y_column)

        ax.scatter(
            x=x_values,
            y=y_values,
            c=colors,
            **scatter_kwargs,
        )

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    @classmethod
    def rank_median_plot(
        cls,
        data: ad.AnnData,
        layer: str = "X",
        y_log: bool = True,
        color_column: str | None = None,
        ax: plt.Axes | None = None,
        scatter_kwargs: dict | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """Plot the ranked protein median intensities across all samples using the scatter method

        Parameters
        ----------
        data : ad.AnnData
            AnnData to plot.
        layer : str
            The AnnData layer to calculate the median value (intensities) across sample. Default is "X"
        y_log : bool
            Log-transform the y-axis. By default True.
        color_column : str, optional
            Column in data.var to use for color encoding. By default None.
        ax : plt.Axes, optional
            Matplotlib axes object to plot on, if None a new figure is created. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        if not layer == "X" and layer not in data.layers:
            raise ValueError(f"Layer {layer} not found in AnnData object")

        # Extract values from the specified layer
        values = np.array(data.X, dtype=np.float64) if layer == "X" else np.array(data.layers[layer], dtype=np.float64)

        # Calculate the median for each protein across all samples
        medians = np.nanmedian(values, axis=0)
        ranked_order = np.argsort(medians)
        ranked_medians = medians[ranked_order]
        proteins = data.var_names[ranked_order] # TODO: add text display option for protein names
        ranks = np.arange(1, len(ranked_medians) + 1)

        # Create a DataFrame from proteins, ranks, and ranked_medians
        ranked_medians_df = pd.DataFrame({
            "Protein": proteins,
            "Rank": ranks,
            "Median": ranked_medians
        })

        # Get the (optional) color values for the proteins
        color_column_for_scatter = None
        if color_column and color_column in data.var.columns:
            colors = _adata_column_to_array(data.var, color_column)
            colors = colors[ranked_order]
            ranked_medians_df["Color"] = colors
            color_column_for_scatter = "Color"

        # Use the scatter method to create the rank plot
        cls.scatter(
            data=ranked_medians_df,
            x_column="Rank",
            y_column="Median",
            color_column=color_column_for_scatter,
            ax=ax,
            scatter_kwargs=scatter_kwargs,
            xlim=xlim,
            ylim=ylim,
        )

        if y_log:
            ax.set_yscale("log")

        ax.set_xlabel("Protein Rank")
        ax.set_ylabel("Median Intensity")
        ax.set_title("Dynamic Range of Protein Median Intensities")
