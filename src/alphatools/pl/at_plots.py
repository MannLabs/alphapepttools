# Plotting functionalities of AlphaTools

import logging
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pandas.api.types import is_numeric_dtype

from alphatools.pl import utils
from alphatools.pl.at_colors import BaseColors, BasePalettes
from alphatools.pp.data import _adata_column_to_array

# logging configuration
logging.basicConfig(level=logging.INFO)

config_file = Path(Path(__file__).parent, "plot_config.yaml")
config = utils.load_plot_config(config_file)


def add_vline(
    ax: plt.Axes,
    x: float | list[float | int],
    color: str = "black",
    linestyle: str = "--",
    linewidth: float = 1,
) -> None:
    """Add a vertical line to a matplotlib axes object"""
    if isinstance(x, float | int):
        ax.axvline(x=x, color=color, linestyle=linestyle, linewidth=linewidth)
    elif isinstance(x, list):
        for xi in x:
            ax.axvline(x=xi, color=color, linestyle=linestyle, linewidth=linewidth)


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


def _make_legend(
    ax: plt.Axes,
    patches: list[mpl.patches.Patch],
    **kwargs,
) -> None:
    """Add a legend to a matplotlib axes object"""
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
        _make_legend(ax, patches, **legend_kwargs)
    elif isinstance(legend, mpl.legend.Legend):
        try:
            ax.add_artist(legend)
        except Exception:
            logging.exception("Error adding legend. Ignoring legend.")
    elif legend:
        logging.warning("Invalid legend parameter. Ignoring legend.")


class Plots:
    """Class for creating figures with matplotlib

    Basic configuration for matplotlib plots is loaded from a YAML file
    and set to generate consistent plots.

    """

    def __init__(
        self,
        config_file: str = "plot_config.yaml",
    ):
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file {config_file} not found")

        config = utils.load_plot_config(config_file)

        self.fontfamily = config["font_family"]
        self.font = config["default_font"]

        self.fontsize_small = config["font_sizes"]["small"]
        self.fontsize_medium = config["font_sizes"]["medium"]
        self.fontsize_large = config["font_sizes"]["large"]

        self.legend_size = config["legend"]["legend_size"]

        self.marker_size_small = config["marker_sizes"]["small"]
        self.marker_size_medium = config["marker_sizes"]["medium"]
        self.marker_size_large = config["marker_sizes"]["large"]

        self.qualitative_colorscale = config["colorscales"]["qualitative"]
        self.sequential_colorscale = config["colorscales"]["sequential"]
        self.diverging_colorscale = config["colorscales"]["diverging"]

        self.hi_color = config["highlight_colors"]["high"]
        self.lo_color = config["highlight_colors"]["low"]
        self.highlight_color = config["highlight_colors"]["general"]

    @classmethod
    def histogram(
        cls,
        data: pd.DataFrame | ad.AnnData,
        value_column: str,
        color_column: str | None = None,
        bins: int = 10,
        color: str = "blue",
        ax: plt.Axes | None = None,
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
            _, ax = plt.subplots(1, 1)

        values = _adata_column_to_array(data, value_column)
        if not is_numeric_dtype(values):
            raise ValueError("Value column must contain numeric data")

        if color_column is None:
            color = BaseColors.get(color)
            ax.hist(values, bins=bins, color=color, **hist_kwargs)

        if color_column is not None:
            colors = _adata_column_to_array(data, color_column)

            levels = np.unique(colors)

            if palette is None:
                palette = BasePalettes.get("qualitative", n=len(levels))

            for _color, level in zip(palette, levels, strict=False):
                ax.hist(values[colors == level], bins=bins, color=_color, **hist_kwargs)

            _parse_legend(ax, legend, palette, levels, **legend_kwargs)
