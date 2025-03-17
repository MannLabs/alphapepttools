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
from alphatools.pl.colors import BaseColormaps, BaseColors, BasePalettes
from alphatools.pl.figure import create_figure
from alphatools.pp.data import _adata_column_to_array

# logging configuration
logging.basicConfig(level=logging.INFO)

config = defaults.plot_settings.to_dict()


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


def _parse_legend(
    ax: plt.Axes,
    legend: str | mpl.legend.Legend | None,
    palette: list[tuple] | None,
    levels: list[str] | None,
    **legend_kwargs,
) -> None:
    """Parse the legend parameter of a plot method. Either create a legend or try to add the provided one"""
    if legend == "auto":
        patches = _make_legend_patches(palette, levels)
        make_legend(ax, patches, **legend_kwargs)
    elif isinstance(legend, mpl.legend.Legend):
        try:
            ax.add_artist(legend)
        except Exception:
            logging.exception("Error adding legend. Ignoring legend.")
    elif legend:
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

            _parse_legend(ax, legend, palette, levels, **legend_kwargs)

    @classmethod
    def scatter(  # noqa: C901 TODO: Refactor into smaller functions & simplify
        cls,
        data: pd.DataFrame | ad.AnnData,
        x_column: str,
        y_column: str,
        color_column: str | None = None,
        color: str = "blue",
        ax: plt.Axes | None = None,
        palette: list[tuple] | None = None,
        legend: str | mpl.legend.Legend | None = None,
        scatter_kwargs: dict | None = None,
        legend_kwargs: dict | None = None,
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
            Column in data to use for color encoding. Overrides color parameter. By default None.
        color : str, optional
            Color to use for the scatterplot. By default "blue".
        ax : plt.Axes, optional
            Matplotlib axes object to plot on, if None a new figure is created. By default None.
        palette : list[tuple], optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
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

        # Avoid overplotting legend until gradient fill Patch is implemented
        override_legend = False

        if not ax:
            _, ax = create_figure()

        x_values = _adata_column_to_array(data, x_column)
        if not is_numeric_dtype(x_values):
            raise ValueError("X column must contain numeric data")
        y_values = _adata_column_to_array(data, y_column)
        if not is_numeric_dtype(y_values):
            raise ValueError("Y column must contain numeric data")

        if color_column is None:
            # TODO: Handle this better, e.g. eliminating ".get()" and making base colors a dict, which would allow for better dict.get(key, alternative) syntax
            try:
                color = BaseColors.get(color)
            except TypeError:
                print(f"Color {color} not found in base colors. Using {color} directly.", flush=True)

            ax.scatter(x_values, y_values, color=color, **scatter_kwargs)

        if color_column is not None:
            colors = _adata_column_to_array(data, color_column)
            color_levels = np.unique(colors)

            if palette is None:
                palette = BasePalettes.get("qualitative", n=len(color_levels))
                if len(set(palette)) < len(color_levels):
                    logging.info(
                        "Scatterplot got more levels than colors in qualitative palette. Switching to sequential colormap."
                    )
                    palette = BaseColormaps.get("sequential")(np.linspace(0, 1, len(color_levels)))

                    # TODO: Add gradient patch from smallest to largest value in color_levels and generate legend
                    override_legend = True

            for _color, color_level in zip(palette, color_levels, strict=False):
                ax.scatter(
                    x_values[colors == color_level], y_values[colors == color_level], color=_color, **scatter_kwargs
                )

            if not override_legend:
                _parse_legend(ax, legend, palette, color_levels, **legend_kwargs)

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
