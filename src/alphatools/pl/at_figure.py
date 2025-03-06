from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alphatools.pl import at_colors, defaults

config = defaults.plot_settings.to_dict()


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


def _indexable_axes(
    axs: plt.Axes | list[plt.Axes] | np.ndarray,
) -> np.ndarray:
    if not isinstance(axs, plt.Axes | list | np.ndarray):
        raise TypeError("Invalid axes provided")

    if isinstance(axs, plt.Axes):
        axs = np.array([[axs]], dtype=object)
    elif isinstance(axs, list):
        axs = np.array(axs, dtype=object)

    if isinstance(axs, np.ndarray) and axs.ndim == 1:
        axs = np.expand_dims(axs, axis=0)

    return axs


class AxisManager:
    """Manage axes objects and make them iterable. Apply consistent styling."""

    def __init__(
        self,
        axs: plt.Axes | list[plt.Axes],
    ):
        axs = _indexable_axes(axs)

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


def _parse_figsize(
    figsize: tuple[float, float] | tuple[str, str] | None,
) -> tuple[float, float]:
    """Allow for figsize to be a tuple of strings to access valid presets from the config file"""
    valid_preset_sizes = config["preset_sizes"]

    if figsize is None:
        figsize = (valid_preset_sizes["1"] / 25.4, valid_preset_sizes["1"] / 25.4)
    elif isinstance(figsize[0], str) and isinstance(figsize[1], str):
        figsize = (valid_preset_sizes[figsize[0]] / 25.4, valid_preset_sizes[figsize[1]] / 25.4)
    elif isinstance(figsize[0], int) and isinstance(figsize[1], int):
        figsize = (figsize[0], figsize[1])
    else:
        raise ValueError(
            "Invalid figsize provided. Must be either a tuple of strings to access valid presets from the config or a tuple of integers"
        )

    return figsize


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | tuple[str, str] | None = None,
    height_ratios: list[float] | None = None,
    width_ratios: list[float] | None = None,
    figure_padding: float | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a figure with a specified number of rows and columns

    Parameters
    ----------
    nrows : int, optional
        The number of rows in the figure, by default 1

    ncols : int, optional
        The number of columns in the figure, by default 1

    figsize : tuple[float, float] | tuple[str, str], optional
        The size of the figure in inches. If a tuple of strings is provided, the strings must be valid keys in the config file, by default None

    height_ratios : list[float], optional
        The height ratios of the rows in the figure, by default None

    width_ratios : list[float], optional
        The width ratios of the columns in the figure, by default None

    figure_padding : float, optional
        The margin padding to apply to the figure, by default None

    Returns
    -------
    fig : plt.Figure
        The figure object

    axs : list[plt.Axes]
        A 2D array of axes objects

    """
    # parse figure size, fall back to defaults if none is given
    figsize = _parse_figsize(figsize)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios},
    )

    fig.patch.set_facecolor("white")

    if figure_padding is not None:
        fig.tight_layout(pad=figure_padding)

    return fig, _indexable_axes(axs)


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: str,
    dpi: int | None = None,
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

    figure_padding : float, optional
        The margin padding to apply to the figure, by default 3

    transparent : bool, optional
        Whether to save a .png figure with a transparent background.

    **kwargs
        Additional keyword arguments to pass to fig.savefig

    """
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".png"):
        filename += ".png"

    if dpi is None:
        dpi = config["resolution"]["dpi"]

    fig.savefig(
        Path(output_dir) / filename,
        dpi=dpi,
        transparent=transparent,
        **kwargs,
    )
