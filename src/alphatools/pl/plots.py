# Plotting functionalities of AlphaTools

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from matplotlib.patches import Rectangle

# logging configuration
logging.basicConfig(level=logging.INFO)


def load_plot_config(config_path: str = "plot_config.yaml") -> dict:
    """Load the plot configuration file"""
    with Path.open(config_path) as file:
        return yaml.safe_load(file)


def _show_rgba_color_list(colors: list) -> None:
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.imshow([colors], aspect="auto")
    ax.axis("off")
    plt.show()


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

        config = load_plot_config(config_file)

        self.fontfamily = config["font_family"]
        self.font = config["default_font"]

        self.fontsize_small = config["font_sizes"]["small"]
        self.fontsize_medium = config["font_sizes"]["medium"]
        self.fontsize_large = config["font_sizes"]["large"]

        self.title_size = config["axes"]["title_size"]
        self.label_size = config["axes"]["label_size"]
        self.tick_size = config["axes"]["tick_size"]

        self.legend_size = config["legend"]["legend_size"]

        self.marker_size_small = config["marker_sizes"]["small"]
        self.marker_size_medium = config["marker_sizes"]["medium"]
        self.marker_size_large = config["marker_sizes"]["large"]

        self.linewidth_small = config["linewidths"]["linewidth_small"]
        self.linewidth_medium = config["linewidths"]["linewidth_medium"]
        self.linewidth_large = config["linewidths"]["linewidth_large"]

        self.qualitative_colorscale = config["colorscales"]["qualitative"]
        self.sequential_colorscale = config["colorscales"]["sequential"]
        self.diverging_colorscale = config["colorscales"]["diverging"]

        self.hi_color = config["highlight_colors"]["high"]
        self.lo_color = config["highlight_colors"]["low"]
        self.highlight_color = config["highlight_colors"]["general"]

        # assign colorscales
        self.custom_colorscales = {
            "rspectral": self.rearranged_spectral_colorscale(),
        }

        # set global rcParams
        plt.rcParams.update(
            {
                "pdf.fonttype": 42,
                "font.family": self.fontfamily,
                "font.sans-serif": self.font,
                "font.size": self.fontsize_medium,
                "axes.titlesize": self.fontsize_medium,
                "axes.labelsize": self.fontsize_medium,
                "xtick.labelsize": self.fontsize_medium,
                "ytick.labelsize": self.fontsize_medium,
                "legend.fontsize": self.fontsize_medium,
                "lines.linewidth": self.linewidth_medium,
                "lines.markersize": self.marker_size_medium,
            }
        )

    @staticmethod
    def show_color(
        colorhex: str,
    ) -> None:
        """Show a color in a matplotlib plot"""
        _, ax = plt.subplots()
        ax.add_patch(Rectangle((0, 0), 1, 1, color=colorhex))
        ax.axis("off")
        plt.show()

    def get_palette(
        self,
        palette_name: str,
        n_colors: int = 10,
    ) -> list:
        """Get a color palette, either from the custom colorscales or from seaborn"""
        if palette_name in self.special_colorscales:
            return sns.color_palette(self.custom_colorscales[palette_name], n_colors)
        return sns.color_palette(palette_name, n_colors)

    def _rearranged_spectral_colorscale(
        self,
    ) -> list:
        """Rearrange the spectral colorscale for better separability"""
        _pal = sns.color_palette("Spectral", 11).as_hex()
        return [_pal[i] for i in [0, 9, 10, 3, 5, 1, 8, 2, 7]]

    def _set_figure_font_sizes(
        self,
    ) -> None:
        """Set font sizes for matplotlib"""
        plt.rc("axes", titlesize=self.fontsize_medium)
        plt.rc("axes", labelsize=self.fontsize_medium)
        plt.rc("xtick", labelsize=self.fontsize_medium)
        plt.rc("ytick", labelsize=self.fontsize_medium)
        plt.rc("legend", fontsize=self.fontsize_medium)
        plt.rc("figure", titlesize=self.fontsize_medium)

    def _set_line_widths(
        self,
        ax: plt.Axes,
    ) -> None:
        """Set line widths for matplotlib"""
        for spine in ax.spines.values():
            spine.set_linewidth(self.linewidth_medium)
        ax.tick_params(width=self.linewidth_medium)


def histogram() -> None:
    """Plot a histogram of the data"""
    raise NotImplementedError
