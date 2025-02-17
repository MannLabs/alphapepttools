# Plotting functionalities of AlphaTools

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from alphatools.pl import utils
from alphatools.pl.at_colors import BaseColors, BasePalettes

# logging configuration
logging.basicConfig(level=logging.INFO)

config_file = Path(Path(__file__).parent, "plot_config.yaml")
config = utils.load_plot_config(config_file)


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

    @staticmethod
    def _add_vline(
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

    @staticmethod
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

    @staticmethod
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

    @classmethod
    def _parse_legend(
        cls,
        ax: plt.Axes,
        legend: str | mpl.legend.Legend | None,
        palette: list[tuple] | None,
        levels: list[str] | None,
        **legend_kwargs,
    ) -> None:
        """Parse the legend parameter of a plot method. Either create a legend or try to add the provided one"""
        if legend == "auto":
            patches = cls._make_legend_patches(palette, levels)
            cls._make_legend(ax, patches, **legend_kwargs)
        elif isinstance(legend, mpl.legend.Legend):
            try:
                ax.add_artist(legend)
            except Exception:
                logging.exception("Error adding legend. Ignoring legend.")
        elif legend:
            logging.warning("Invalid legend parameter. Ignoring legend.")

    @classmethod
    def histogram(
        cls,
        data: list | np.ndarray | pd.Series,
        color_data: list | np.ndarray | pd.Series | None = None,
        bins: int = 10,
        color: str = "blue",
        ax: plt.Axes | None = None,
        palette: list[tuple] | None = None,
        legend: str | mpl.legend.Legend | None = None,
        hist_kwargs: dict | None = None,
        legend_kwargs: dict | None = None,
    ) -> None:
        """Plot a histogram of a list, an array or a pandas Series"""
        hist_kwargs = hist_kwargs or {}
        legend_kwargs = legend_kwargs or {}

        if not ax:
            fig, ax = plt.subplots(1, 1)

        if color_data is None:
            color = BaseColors.get(color)
            ax.hist(data, bins=bins, color=color, **hist_kwargs)

        if color_data is not None:
            levels = np.unique(color_data)

            if palette is None:
                palette = BasePalettes.get("qualitative", n=len(levels))

            if len(color_data) != len(data):
                raise ValueError("Data and color data must have the same length")

            for _color, level in zip(palette, levels, strict=False):
                ax.hist(data[color_data == level], bins=bins, color=_color, **hist_kwargs)

            cls._parse_legend(ax, legend, palette, levels, **legend_kwargs)
