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
    """Apply alphapepttools style to a matplotlib axes object

    Parameters
    ----------
    ax
        Matplotlib axes object to style

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        The styled axes object

    Example
    -------
    .. code-block:: python

        fig, ax = plt.subplots()
        stylize(ax)
    """
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
    ax
        The axes object to apply labels to
    xlabel
        The x-axis label. If `None`, existing label is not changed
    ylabel
        The y-axis label. If `None`, existing label is not changed
    title
        The title of the axes. If `None`, existing title is not changed
    label_parser
        Function to parse labels from computation to presentation context,
        e.g., `upregulated_proteins` -> "Upregulated Proteins"
    enumeration
        String to enumerate the plot in the top left, e.g., "A", "B", "C"
    enumeration_xytext
        Offset of enumeration text in points relative to top left of axis.
        Does not scale with resolution or plot size

    Example
    -------
    .. code-block:: python

        import string
        from alphapepttools.pl import label_axes

        fig, axs = plt.subplots(2, 2)
        for i, ax in enumerate(axs.flat):
            label_axes(
                ax,
                xlabel="X values",
                ylabel="Y values",
                title=f"Function {i + 1}",
                enumeration=string.ascii_uppercase[i],
            )
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
    """Create a figure with consistent styling and an AxisManager for subplot iteration

    Creates matplotlib figures where the exported file matches exactly what you see
    in Jupyter notebooks, eliminating iterative size/padding adjustments.

    Parameters
    ----------
    nrows
        Number of rows in the figure
    ncols
        Number of columns in the figure
    figsize
        Figure size in inches or as config preset keys (e.g., `("1", "2")`)
    height_ratios
        Height ratios of the rows
    width_ratios
        Width ratios of the columns
    subplots_kwargs
        Additional keyword arguments for `plt.subplots`
    gridspec_kwargs
        Additional keyword arguments for gridspec

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure object
    :class:`AxisManager`
        An iterable and indexable manager for accessing styled axes

    Examples
    --------
    Basic usage with iteration:

    .. code-block:: python

        from alphapepttools.pl import create_figure

        fig, axm = create_figure(1, 2, figsize=(7, 3))

        # Iterate through axes
        for ax in [axm.next(), axm.next()]:
            ax.plot([1, 2, 3], [1, 4, 9])

    Using indexing:

    .. code-block:: python

        fig, axm = create_figure(2, 2, figsize=(6, 6))

        # Access by index
        axm[0].scatter(x, y1)
        axm[1].scatter(x, y2)

        # Or by row/column
        axm[1, 0].scatter(x, y3)
        axm[1, 1].scatter(x, y4)
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
    """Save a figure in publication-friendly format

    Parameters
    ----------
    fig
        The figure to save
    filename
        Filename with extension. Defaults to `.png` if no extension given
    output_dir
        Output directory. Created if it doesn't exist
    dpi
        Resolution of the figure, defaults from config
    transparent
        Whether to save with transparent background (PNG only)
    **kwargs
        Additional keyword arguments passed to `fig.savefig`

    Example
    -------
    .. code-block:: python

        from alphapepttools.pl import create_figure, save_figure

        fig, axm = create_figure(1, 2, figsize=(7, 3))
        # ... add plots ...
        save_figure(fig, "my_plot.png", "./figures", dpi=300)
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
