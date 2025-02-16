from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alphatools.pl import at_colors, utils

config_file = Path(Path(__file__).parent, "plot_config.yaml")
config = utils.load_plot_config(config_file)


# Adapted from https://github.com/ersilia-os/stylia.git
def stylize(
    ax: plt.Axes,
) -> plt.Axes:
    """Apply AlphaTools style to a matplotlib axes object"""
    ax.set_prop_cycle("color", at_colors.BasePalettes.get("qualitative"))
    ax.grid(visible=True, linewidth=config["linewidths"]["small"])
    ax.xaxis.set_tick_params(width=config["linewidths"]["small"], labelsize=config["axes"]["tick_size"])
    ax.yaxis.set_tick_params(width=config["linewidths"]["small"], labelsize=config["axes"]["tick_size"])
    return ax


def label(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Apply labels to a matplotlib axes object"""
    ax.set_xlabel(xlabel, fontsize=config["axes"]["label_size"]) if xlabel is not None else ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=config["axes"]["label_size"]) if ylabel is not None else ax.set_ylabel("")
    ax.set_title(title, fontsize=config["axes"]["title_size"]) if title is not None else ax.set_title("")
    return ax


class AxisManager:
    """Manage axes objects and make them iterable. Apply consistent styling."""

    def __init__(
        self,
        axs: plt.Axes | list[plt.Axes],
    ):
        if axs is None or not len(axs):
            raise ValueError("No axes provided")
        if not isinstance(axs, plt.Axes | list | np.ndarray):
            raise TypeError("Invalid axes provided")

        if isinstance(axs, plt.Axes):
            axs = np.array([[axs]], dtype=object)
        elif isinstance(axs, list):
            axs = np.array(axs, dtype=object)
            if axs.ndim == 1:
                axs = np.expand_dims(axs, axis=0)

        self.axs = axs
        self.axs_flat = axs.flatten()

        self.current_i = 0
        self.rows, self.cols = self.axs.shape

    def __getitem__(
        self,
        key: int | tuple[int, int],
    ):
        if isinstance(key, int):
            i = key
            if key >= len(self.axs_flat):
                raise IndexError(f"Axes index {i} out of bounds")
            ax = self.axs_flat[i]
            self.current_i = i + 1
        elif isinstance(key, tuple):
            i, j = key
            if i >= self.rows or j >= self.cols:
                raise IndexError(f"Axes index {i}, {j} out of bounds")
            ax = self.axs[i, j]
            # standard row-major indexing
            self.current_i = i * self.cols + j
        return stylize(ax)

    def restart(self) -> None:
        """Reset the current index of AxisManager to 0"""
        self.current_i = 0

    def next(self) -> plt.Axes:
        """Get the next axes object in the sequence"""
        if self.current_i >= len(self.axs_flat):
            raise StopIteration("No more axes available")
        ax = self.axs_flat[self.current_i]
        self.current_i += 1
        return stylize(ax)


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: str,
    dpi: int = config["resolution"]["dpi"],
    width_mm: int = 89,
    height_mm: int = 89,
    paper_width: int | None = None,
    paper_height: str | None = None,
    figure_padding: float = 0.1,
    transparent: bool = False,  # noqa: FBT002, FBT001 shadows savefig signature
    **kwargs,
) -> None:
    """Save a figure in a publication friendly format

    Parameters
    ----------
    fig : matplotlib.pyplot.Figure
        The figure to save

    filename : str
        The filename to save the figure. Must have a supported extension.
        If no extension is given, the figure will be saved as a .png file.

    output_dir : str
        The directory to save the figure in

    dpi : int, optional
        The resolution of the figure, by default 300

    width_mm : int, optional
        The width of the figure in millimeters, by default None.

    height_mm : int, optional
        The height of the figure in millimeters, by default None.

    nature_width : str
        Width of the figure in the scientific paper. One unit
        corresponds approximately to one column in a two-column
        publication format. The following values are supported:

        - '2' = 183 mm (7.2 inches)
        - '1.5' = 135 mm (5.3 inches)
        - '1' = 89 mm (3.5 inches)
        - '0.5' = 45 mm (1.8 inches)
        - '0.25' = 22.5 mm (0.9 inches)

        If specified, 'nature_width' overrides width_mm.

    nature_height : str
        Height of the figure in the scientific paper. One unit
        corresponds approximately to one column in a two-column
        publication format. The following values are supported:
        - '2' = 183 mm (7.2 inches)
        - '1.5' = 135 mm (5.3 inches)
        - '1' = 89 mm (3.5 inches)
        - '0.5' = 45 mm (1.8 inches)
        - '0.25' = 22.5 mm (0.9 inches)

        If specified, 'nature_height' overrides height_mm.

    transparent : bool, optional
        Whether to save a .png figure with a transparent background.

    **kwargs
        Additional keyword arguments to pass to fig.savefig

    """
    if not Path(output_dir).exists():
        raise FileNotFoundError(f"Output directory {output_dir} does not exist")

    if not filename.endswith(".png"):
        filename += ".png"

    paper_sizes = {
        "2": config["preset_sizes"]["2"],
        "1.5": config["preset_sizes"]["1.5"],
        "1": config["preset_sizes"]["1"],
        "0.5": config["preset_sizes"]["0.5"],
        "0.25": config["preset_sizes"]["0.25"],
    }

    width_in = paper_sizes.get(paper_width, width_mm) / 25.4
    height_in = paper_sizes.get(paper_height, height_mm) / 25.4

    fig.tight_layout(pad=figure_padding)

    fig.set_size_inches(width_in, height_in)

    fig.savefig(
        Path(output_dir) / filename,
        dpi=dpi,
        transparent=transparent,
        **kwargs,
    )
