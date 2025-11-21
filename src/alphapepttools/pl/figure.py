# figure.py

# Defines how matplotlib figures and axes are handled. Main functions are stylize() and label(),
# which apply a consistent layout and appropriately sized labels (based on plot_config.yaml).
# This module also contains functions for creating and saving figures based on matplotlib's "subplots()" method.
# Plotting is handled by the "AxisManager" class, which allows for easy (!) iteration or indexing of subplots,
# while applying consistent styling (see 02_plotting.ipynb for examples).

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alphapepttools.pl import colors, defaults

config = defaults.plot_settings.to_dict()


# Adapted from https://github.com/ersilia-os/stylia.git
def stylize(
    ax: plt.Axes,
) -> plt.Axes:
    """Apply alphapepttools style to a matplotlib axes object"""
    ax.set_prop_cycle("color", colors.BasePalettes.get("qualitative"))
    ax.grid(visible=True, linewidth=config["linewidths"]["small"])
    ax.xaxis.set_tick_params(width=config["linewidths"]["small"], labelsize=config["axes"]["tick_size"])
    ax.yaxis.set_tick_params(width=config["linewidths"]["small"], labelsize=config["axes"]["tick_size"])
    return ax


def label_axes(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    label_parser: Callable | None = None,
    enumeration: str | None = None,
    enumeration_xytext: tuple[float, float] = (-10, 10),
) -> None:
    """Apply labels to a matplotlib axes object

    Parameters
    ----------
    ax : plt.Axes
        The axes object to apply labels to
    xlabel : str, optional
        The x-axis label, by default None (existing label is not changed)
    ylabel : str, optional
        The y-axis label, by default None (existing label is not changed)
    title : str, optional
        The title of the axes, by default None (existing title is not changed)
    label_parser : Callable, optional
        A function to parse labels, by default None. This is useful to convert
        labels from a computation-context to presentation context, e.g. a column
        like upregulated_proteins could be shown as "Upregulated Proteins" in the plot.
    enumeration : str, optional
        A string to enumerate the plot in the top left, e.g. "A", "B", "C", etc.
    enumeration_xytext : Tuple[float, float], optional, by default (-10, 10)
        This parameter describes the offset of the enumeration text in typographic points
        relative to the top left of the axis. This does not scale with resolution or plot
        size, but can be adapted to fit the plot.

    Returns
    -------
    None

    """
    label_parser = label_parser or (lambda x: x)

    ax.set_xlabel(label_parser(xlabel), fontsize=config["axes"]["label_size"]) if xlabel is not None else None
    ax.set_ylabel(label_parser(ylabel), fontsize=config["axes"]["label_size"]) if ylabel is not None else None
    ax.set_title(label_parser(title), fontsize=config["axes"]["title_size"]) if title is not None else None

    # Optionally, add the numeration to the plot
    if enumeration is not None:
        ax.annotate(
            str(enumeration),
            xy=(0, 1),  # This is the anchor: top left of the plot
            xytext=enumeration_xytext,  # This is the text position relative to the offset
            xycoords="axes fraction",  # This tells mpl that the coordinates are relative to the axes
            textcoords="offset points",  # This tells mpl that the text position is in points relative to the anchor
            fontsize=config["font_sizes"]["large"],
            ha="right",
        )


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
        self.current_i = 0
        self.rows, self.cols = self.axs.shape

    @property
    def _axs_flat(self) -> np.ndarray:
        return self.axs.flatten()

    def __getitem__(
        self,
        key: int | tuple[int, int],
    ):
        if isinstance(key, int):
            i = key
            if i >= len(self._axs_flat):
                raise IndexError(f"Axes index {i} out of bounds")
            ax = self._axs_flat[i]
            self.current_i = i + 1
        elif isinstance(key, tuple):
            i, j = key
            if i >= self.rows or j >= self.cols:
                raise IndexError(f"Axes index {i}, {j} out of bounds")
            ax = self.axs[i, j]
            # standard row-major indexing
            self.current_i = i * self.cols + j
        return stylize(ax)

    def reset(self) -> None:
        """Reset the current index of AxisManager to 0"""
        self.current_i = 0

    def next(self) -> plt.Axes:
        """Get the next axes object in the sequence"""
        if self.current_i >= len(self._axs_flat):
            raise StopIteration("No more axes available")
        ax = self._axs_flat[self.current_i]
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
        try:
            figsize = (valid_preset_sizes[figsize[0]] / 25.4, valid_preset_sizes[figsize[1]] / 25.4)
        except KeyError as e:
            raise ValueError(
                f"Invalid strings {figsize[0]} and {figsize[1]} provided. Must be valid keys in the config file"
            ) from e
    elif isinstance(figsize[0], int | float) and isinstance(figsize[1], int | float):
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
    subplots_kwargs: dict | None = None,
    gridspec_kwargs: dict | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a figure with a specified number of rows and columns. Returns an AxisManager object to manage axes objects.

    This is especially useful for creating subplots with consistent styling. Importantly, it should completely sync the
    plot a user sees in a jupyter notebook with the exported (e.g. .png) figure file. The aim of this is to entirely
    eliminate time consuming iterations of checking the exported plot and going back to adjust sizes/padding in the code.

    Example:

    # This works:
    fig, axm = create_figure(nrows=2, ncols=2, figsize = (4, 4))
    x = np.linspace(0, 10, 100)
    functions = [lambda x: np.sin(x + 1), lambda x: np.sin(x) * 2, lambda x: np.sin(x) + 2, lambda x: np.sin(x) - 2]
    for i, func in enumerate(functions):
        ax = axm[i]
        ax.scatter(x, func(x))

    # Just the same as this:
    fig, axm = create_figure(nrows=1, ncols=4, figsize = (8, 2))
    x = np.linspace(0, 10, 100)
    functions = [lambda x: np.sin(x + 1), lambda x: np.sin(x) * 2, lambda x: np.sin(x) + 2, lambda x: np.sin(x) - 2]
    for i, func in enumerate(functions):
        ax = axm[i]
        ax.scatter(x, func(x))

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

    Returns
    -------
    fig : plt.Figure
        The figure object

    axm : AxisManager object
        An iterable and indexable object to manage matplotlib.axes objects

    """
    # set global rcParams
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": config["font_family"],
            "font.sans-serif": config["default_font"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # parse figure size, fall back to defaults if none is given
    figsize = _parse_figsize(figsize)

    # Handle special parameters for subplots and gridspecs for more complex plots
    subplots_kwargs = {"constrained_layout": True, **(subplots_kwargs or {})}
    gridspec_kwargs = {"width_ratios": width_ratios, "height_ratios": height_ratios, **(gridspec_kwargs or {})}

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        gridspec_kw=gridspec_kwargs,
        **subplots_kwargs,
    )

    fig.patch.set_facecolor("white")

    return fig, AxisManager(axs)


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
        The directory to save the figure in. Will be created in case it does not exist.

    dpi : int, optional
        The resolution of the figure, taken by default from config

    transparent : bool, optional
        Whether to save a .png figure with a transparent background.

    **kwargs
        Additional keyword arguments to pass to fig.savefig

    """
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if dpi is None:
        dpi = config["resolution"]["dpi"]

    fig.savefig(
        Path(output_dir) / filename,
        dpi=dpi,
        transparent=transparent,
        **kwargs,
    )
