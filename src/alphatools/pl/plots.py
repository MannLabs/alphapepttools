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

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from alphatools.pl import defaults
from alphatools.pl.colors import BaseColors, BasePalettes, get_color_mapping
from alphatools.pl.figure import create_figure, label_axes
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
    color_dict: dict[str, str | tuple],
) -> list[mpl.patches.Patch]:
    """Create legend patches for a matplotlib legend from a value-to-color mapping"""
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
    palette: list[str | tuple] | None,
    legend: str | mpl.legend.Legend | None = None,
    **legend_kwargs,
) -> None:
    """Add a legend to an axis object.

    If levels and palette are provided and legend is None, a legend is created automatically.
    if legend is set to "auto", a legend is created from the levels and a default palette.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to add the legend to.
    levels : list[str] | None
        List of levels to use for the legend. Duplicates are removed.
    palette : list[str | tuple] | None
        List of colors to use for the legend. If None, a default palette will be used. By default None.

    """
    if legend not in ["auto", None]:
        raise ValueError("legend must be 'auto' or None")

    if levels is None:
        logging.warning("No levels provided. Skipping legend creation.")
        return

    levels = np.unique(levels)

    # Determine palette, i.e. list of colors to show in the legend
    if palette is None:
        if legend == "auto":
            palette = BasePalettes.get("qualitative")
        else:
            raise ValueError("Palette must be provided if legend is not set to 'auto'")

    color_dict = get_color_mapping(levels, palette)
    patches = _make_legend_patches(color_dict)
    make_legend(ax, patches, **legend_kwargs)


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
    label_kwargs = {"fontsize": config["font_sizes"]["medium"], **(label_kwargs or {})}
    line_kwargs = {"color": BaseColors.get("black"), "linewidth": config["linewidths"]["medium"], **(line_kwargs or {})}
    label_parser = label_parser or (lambda x: x)

    if not len(x_values) == len(y_values) == len(labels):
        raise ValueError("x_values, y_values, and labels must have the same length")

    # convert to numpy arrays for consistency & remove any nans
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    labels = np.array(labels)

    keep_mask = np.logical_or(np.isnan(x_values), np.isnan(y_values))
    x_values = x_values[~keep_mask]
    y_values = y_values[~keep_mask]
    labels = labels[~keep_mask]

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
        label_spacing_display = fontsize_display * 2
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
    # lines = sorted(lines, key=lambda line: line[1][1], reverse=True)

    for line in lines:
        ax.plot(line[0], line[1], **line_kwargs)
        if x_anchors is not None:
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

        values = _adata_column_to_array(data, value_column)

        if color_column is None:
            color = BaseColors.get(color)
            ax.hist(values, bins=bins, color=color, **hist_kwargs)
        else:
            color_values = _adata_column_to_array(data, color_column)
            palette = palette or BasePalettes.get("qualitative")
            color_dict = color_dict or get_color_mapping(color_values, palette)
            missing = set(np.unique(color_values)) - set(color_dict)
            for level in missing:
                color_dict[level] = BaseColors.get("grey")

            for level, level_color in color_dict.items():
                ax.hist(
                    values[color_values == level],
                    bins=bins,
                    color=level_color,
                    **hist_kwargs,
                )

            if legend is not None:
                add_legend(
                    ax=ax,
                    levels=list(color_dict.keys()),
                    palette=list(color_dict.values()),
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
        y_column: str,
        x_column: str,
        color: str = "blue",
        color_column: str | None = None,
        color_map_column: str | None = None,
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
        color_column : str, optional
            Column in data to plot the colors. This must contain actual color values (RGBA, hex, etc.). Overrides color and color_map_column parameters. By default None.
        color_map_column : str, optional
            Column in data to use for color encoding. Overrides color parameter. By default None.
        ax : plt.Axes, optional
            Matplotlib axes object to plot on, if None a new figure is created. By default None.
        palette : list[str | tuple], optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
        color_dict: dict[str, str | tuple], optional
            Supercedes palette, a dictionary mapping levels to colors. By default None. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
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
        legend_kwargs = legend_kwargs or {}
        DEFAULT_GROUP = "data"

        if ax is None:
            _, axm = create_figure()
            ax = axm.next()

        # Handle color encoding: If there is an actual color column, simply color the points accordingly
        if color_column is not None:
            color_map_column = None
            color_dict = None
            color_values = _adata_column_to_array(data, color_column)
        # If there is a color map column, map its levels to a palette
        elif color_map_column is not None:
            color_levels = _adata_column_to_array(data, color_map_column)
            color_dict = color_dict or get_color_mapping(color_levels, palette or BasePalettes.get("qualitative"))
            missing = set(np.unique(color_levels)) - set(color_dict)
            for level in missing:
                color_dict[level] = BaseColors.get("grey")
            color_values = np.array([color_dict[level] for level in color_levels], dtype=object)
        else:
            color_dict = {DEFAULT_GROUP: BaseColors.get(color)}
            color_values = np.array([color_dict[DEFAULT_GROUP]] * len(data))

        # Handle ordering of plotting arrays by string: order by the frequency of the color column
        counts = Counter([str(cv) for cv in color_values])
        order = np.argsort([counts[str(cv)] for cv in color_values])[::-1]
        x_values = _adata_column_to_array(data, x_column)[order]
        y_values = _adata_column_to_array(data, y_column)[order]
        color_values = np.array(color_values)[order]

        ax.scatter(
            x=x_values,
            y=y_values,
            c=color_values,
            **scatter_kwargs,
        )

        if legend is not None and color_dict is not None:
            add_legend(
                ax=ax,
                levels=list(color_dict.keys()),
                palette=list(color_dict.values()),
                legend=legend,
                **legend_kwargs,
            )

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    @classmethod
    def rank_median_plot(
        cls,
        data: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        layer: str = "X",
        color: str = "blue",
        palette: list[str | tuple] | None = None,
        color_dict: dict[str, str | tuple] | None = None,
        legend: str | mpl.legend.Legend | None = None,
        color_column: str | None = None,
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
        color_column : str, optional
            Column in data.var to use for color coding. By default None.
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
            color_map_column=color_column,
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
