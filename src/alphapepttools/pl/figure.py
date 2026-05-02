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
    """Apply alphapepttools visual style to a matplotlib axes object

    Extendable function to configure a matplotlib axes with alphapepttools's default styling, including
    color palette, grid settings, and tick parameters. This ensures consistent
    visual appearance across all plots by applying predefined color cycles,
    enabling gridlines with appropriate thickness, and setting standardized
    tick sizes and widths.

    Parameters
    ----------
    ax
        The matplotlib axes object to style

    Returns
    -------
    The styled matplotlib axes object (same object as input, modified in place)

    Examples
    --------
    Apply styling to a single plot:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        from alphapepttools.pl.figure import stylize

        fig, ax = plt.subplots(figsize=(6, 4))
        stylize(ax)

        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x), label="sin(x)")
        ax.plot(x, np.cos(x), label="cos(x)")
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.legend()
        plt.show()

    Notes
    -----
    The styling applied includes:
    - Setting the color cycle to alphapepttools's qualitative color palette
    - Enabling gridlines with small line width from configuration
    - Configuring x and y axis tick parameters (width and label size)

    The function modifies the axes in place and also returns it for convenience.

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
    """Apply formatted labels and optional enumeration to a matplotlib axes object

    Sets axis labels, title, and optional subplot enumeration (e.g., A, B, C) with
    consistent formatting from alphapepttools configuration. Labels can be optionally
    processed through a parser function to transform computational names into
    presentation-ready text. Enumeration letters are positioned in the top-left
    corner for multi-panel figures.

    Parameters
    ----------
    ax
        The axes object to apply labels to
    xlabel
        The x-axis label, by default None (existing label is not changed)
    ylabel
        The y-axis label, by default None (existing label is not changed)
    title
        The title of the axes, by default None (existing title is not changed)
    label_parser
        A function to parse labels, by default None. This is useful to convert
        labels from a computation-context to presentation context, e.g. a column
        like upregulated_proteins could be shown as "Upregulated Proteins" in the plot
    enumeration
        A string to enumerate the plot in the top left, e.g. "A", "B", "C", etc.
    enumeration_xytext
        The offset of the enumeration text in typographic points relative to the
        top left of the axis. This does not scale with resolution or plot size,
        but can be adapted to fit the plot. Default is (-10, 10)

    Returns
    -------
    None

    Examples
    --------
    Label a single plot with title and axis labels:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        from alphapepttools.pl.figure import label_axes

        fig, ax = plt.subplots(figsize=(6, 4))

        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x))

        label_axes(ax, xlabel="Time (seconds)", ylabel="Amplitude", title="Sine Wave")
        plt.show()

    Create a multi-panel figure with enumeration:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import string
        from alphapepttools.pl.figure import label_axes

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            x = np.linspace(0, 10, 100)
            ax.plot(x, np.sin(x * (i + 1)))

            label_axes(
                ax,
                xlabel="X values",
                ylabel="Y values",
                title=f"Function {i + 1}",
                enumeration=string.ascii_uppercase[i],
            )

        plt.tight_layout()
        plt.show()

    Use a label parser to format computational names:

    .. code-block:: python

        import matplotlib.pyplot as plt
        from alphapepttools.pl.figure import label_axes


        def format_label(label):
            if label is None:
                return None
            # Convert underscore notation to title case
            return label.replace("_", " ").title()


        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([1, 2, 3], [4, 5, 6])

        label_axes(
            ax,
            xlabel="sample_id",
            ylabel="protein_abundance",
            title="abundance_distribution",
            label_parser=format_label,
        )
        # Results in: "Sample Id", "Protein Abundance", "Abundance Distribution"
        plt.show()

    Notes
    -----
    The function applies labels with font sizes from the global configuration.
    If a label_parser is provided, it will be applied to all labels (xlabel,
    ylabel, and title) before setting them. The enumeration text is positioned
    using matplotlib's annotation system with offset points for precise placement.

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
    """Convert various axes input formats to a consistent 2D numpy array

    Transforms matplotlib axes objects from different input formats (single axes,
    1D lists/arrays, or 2D arrays) into a standardized 2D numpy array. This ensures
    consistent indexing behavior regardless of how the axes were originally created,
    enabling unified row-column access patterns in the AxisManager class.

    Parameters
    ----------
    axs
        Matplotlib axes in any of these formats:
        - Single plt.Axes object
        - List of plt.Axes objects (1D)
        - 1D numpy array of plt.Axes objects
        - 2D numpy array of plt.Axes objects

    Returns
    -------
    A 2D numpy array of axes objects with dtype=object, where even single axes
    are wrapped in a 2x2 array structure for consistent indexing

    Raises
    ------
    TypeError
        If the input is not a plt.Axes, list, or numpy array

    Examples
    --------
    Convert a single axes to 2D array:

    .. code-block:: python

        import matplotlib.pyplot as plt
        from alphapepttools.pl.figure import _indexable_axes

        fig, ax = plt.subplots()
        result = _indexable_axes(ax)
        print(result.shape)  # (1, 1)
        print(result[0, 0] is ax)  # True

    Convert a 1D list of axes to 2D array:

    .. code-block:: python

        import matplotlib.pyplot as plt
        from alphapepttools.pl.figure import _indexable_axes

        fig, axes = plt.subplots(1, 3)  # Creates 1D array
        result = _indexable_axes(axes)
        print(result.shape)  # (1, 3)
        # Now accessible with consistent 2D indexing
        print(result[0, 1])  # Middle subplot

    Notes
    -----
    This utility function is essential for AxisManager's ability to handle
    matplotlib's varying subplot creation patterns. When plt.subplots() is called:
    - With nrows=1, ncols=1: returns a single Axes object
    - With either nrows=1 or ncols=1: returns a 1D array
    - With both > 1: returns a 2D array

    By converting all cases to 2D arrays, AxisManager can provide consistent
    iteration and indexing behavior regardless of the original structure.

    """
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
    """Manage and iterate through matplotlib axes with automatic styling

    Provides a convenient interface for working with matplotlib subplots by offering
    sequential iteration, direct indexing, and automatic application of alphapepttools
    styling. The manager allows iterating through subplots using the `next()` method
    while also supporting direct indexing of individual subplots.
    Each accessed axes is automatically styled using the `stylize()` function.

    Attributes
    ----------
    axs
        2D numpy array of matplotlib axes objects
    current_i
        Current position index for sequential iteration
    rows
        Number of rows in the axes grid
    cols
        Number of columns in the axes grid

    Methods
    -------
    next()
        Get the next axes in sequence with automatic styling
    reset()
        Reset the iteration counter to start from the beginning
    __getitem__(key)
        Access axes by index (int) or row/column tuple with automatic styling

    Examples
    --------
    Sequential iteration through subplots:

    .. code-block:: python

        from alphapepttools.pl.figure import create_figure
        import numpy as np

        fig, axm = create_figure(1, 2)

        # First figure
        ax = axm.next()
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x))
        ax.set_title("Sine Wave")

        # Second figure
        ax = axm.next()
        ax.plot(x, np.cos(x))
        ax.set_title("Cosine Wave")

        plt.show()

    Mix sequential and indexed access:

    .. code-block:: python

        from alphapepttools.pl.figure import create_figure
        import numpy as np

        fig, axm = create_figure(2, 3)

        # Use next() for first two plots
        ax = axm.next()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("Plot 1")

        ax = axm.next()
        ax.scatter([1, 2, 3], [2, 4, 6])
        ax.set_title("Plot 2")

        # Skip to specific position using indexing
        ax = axm[1, 1]  # Row 1, Column 1
        ax.bar([1, 2, 3], [3, 2, 1])
        ax.set_title("Plot at (1,1)")

        # Continue with next() from there
        ax = axm.next()
        ax.hist(np.random.randn(100))
        ax.set_title("Histogram")

        plt.show()

    Process data in a loop with automatic progression:

    .. code-block:: python

        from alphapepttools.pl.figure import create_figure
        import numpy as np

        fig, axm = create_figure(2, 2)

        datasets = [
            ("Dataset A", np.random.randn(100)),
            ("Dataset B", np.random.randn(100) + 2),
            ("Dataset C", np.random.randn(100) * 2),
            ("Dataset D", np.random.randn(100) - 1),
        ]

        for title, data in datasets:
            ax = axm.next()
            ax.hist(data, bins=20, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        plt.show()

    Notes
    -----
    The AxisManager is particularly useful when:
    - Creating multiple similar plots in sequence
    - Working with data processing pipelines where plots are generated iteratively
    - You want automatic styling without calling stylize() manually
    - You need to mix sequential and random access to subplots

    """

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
        """Reset the current index of AxisManager to 0

        Resets the internal counter to start iteration from the beginning.
        Useful when you need to make multiple passes through the same set
        of subplots.

        Examples
        --------
        .. code-block:: python

            fig, axm = create_figure(1, 3)

            # First pass
            for i in range(3):
                ax = axm.next()
                ax.plot([1, 2, 3], [i, i + 1, i + 2])

            # Reset and make second pass
            axm.reset()
            for i in range(3):
                ax = axm.next()
                ax.set_title(f"Plot {i + 1}")

        """
        self.current_i = 0

    def next(self) -> plt.Axes:
        """Get the next axes object in the sequence

        Returns the next subplot in row-major order (left to right, top to bottom)
        and automatically applies alphapepttools styling. Advances the internal
        counter for subsequent calls.

        Returns
        -------
        The next styled matplotlib axes object

        Raises
        ------
        StopIteration
            If no more axes are available in the sequence

        Examples
        --------
        .. code-block:: python

            fig, axm = create_figure(1, 2)

            # Get first subplot
            ax = axm.next()
            ax.plot([1, 2, 3], [1, 4, 9])

            # Get second subplot
            ax = axm.next()
            ax.scatter([1, 2, 3], [2, 4, 6])

        """
        if self.current_i >= len(self._axs_flat):
            raise StopIteration("No more axes available")
        ax = self._axs_flat[self.current_i]
        self.current_i += 1
        return stylize(ax)


def _parse_figsize(
    figsize: tuple[float, float] | tuple[str, str] | None,
) -> tuple[float, float]:
    """Convert figsize input to inches, supporting preset string keys

    Allows figsize to be specified as preset strings (e.g., ("1", "0.5")) that map
    to predefined column widths in mm from the configuration, or as direct numeric
    values in inches. Converts mm presets to inches for matplotlib compatibility.

    Parameters
    ----------
    figsize
        Figure size as (width, height) in inches, preset string keys, or None for defaults

    Returns
    -------
    Tuple of (width, height) in inches

    Raises
    ------
    ValueError
        If string keys are invalid or mixed string/numeric types are provided

    """
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
    """Create a figure with a specified number of rows and columns

    Returns an AxisManager object to manage axes objects. This is especially useful
    for creating subplots with consistent styling. Importantly, it should completely
    sync the plot a user sees in a jupyter notebook with the exported (e.g. .png)
    figure file. The aim of this is to entirely eliminate time consuming iterations
    of checking the exported plot and going back to adjust sizes/padding in the code.

    Parameters
    ----------
    nrows
        The number of rows in the figure, by default 1
    ncols
        The number of columns in the figure, by default 1
    figsize
        The size of the figure in inches. If a tuple of strings is provided, the
        strings must be valid keys in the config file, by default None
    height_ratios
        The height ratios of the rows in the figure, by default None
    width_ratios
        The width ratios of the columns in the figure, by default None
    subplots_kwargs
        Additional keyword arguments to pass to plt.subplots, by default None
    gridspec_kwargs
        Additional keyword arguments for gridspec configuration, by default None

    Returns
    -------
    fig
        The figure object
    axm
        An iterable and indexable AxisManager object to manage matplotlib.axes objects

    Examples
    --------
    Create a 2x2 grid of subplots:

    .. code-block:: python

        import numpy as np
        from alphapepttools.pl.figure import create_figure

        fig, axm = create_figure(nrows=2, ncols=2, figsize=(4, 4))
        x = np.linspace(0, 10, 100)
        functions = [lambda x: np.sin(x + 1), lambda x: np.sin(x) * 2, lambda x: np.sin(x) + 2, lambda x: np.sin(x) - 2]
        for i, func in enumerate(functions):
            ax = axm[i]
            ax.scatter(x, func(x))

    Create the same plots in a 1x4 layout:

    .. code-block:: python

        import numpy as np
        from alphapepttools.pl.figure import create_figure

        fig, axm = create_figure(nrows=1, ncols=4, figsize=(8, 2))
        x = np.linspace(0, 10, 100)
        functions = [lambda x: np.sin(x + 1), lambda x: np.sin(x) * 2, lambda x: np.sin(x) + 2, lambda x: np.sin(x) - 2]
        for i, func in enumerate(functions):
            ax = axm[i]
            ax.scatter(x, func(x))

    """
    # set global rcParams
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
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
    """Save a figure in a publication-friendly format

    Saves matplotlib figures with appropriate settings for scientific publications,
    automatically creating the output directory if it doesn't exist. Uses configured
    DPI settings by default for consistent high-quality output.

    Parameters
    ----------
    fig
        The figure to save
    filename
        The filename to save the figure. Must have a supported extension.
        If no extension is given, the figure will be saved as a .png file
    output_dir
        The directory to save the figure in. Will be created if it does not exist
    dpi
        The resolution of the figure, taken by default from config
    transparent
        Whether to save a .png figure with a transparent background
    **kwargs
        Additional keyword arguments to pass to fig.savefig

    Returns
    -------
    None

    Examples
    --------
    Save a figure with default settings:

    .. code-block:: python

        from alphapepttools.pl.figure import create_figure, save_figure
        import numpy as np

        fig, axm = create_figure(1, 1)
        ax = axm.next()
        ax.plot(np.random.randn(100))

        save_figure(fig, "my_plot.png", "figures/")

    Save with custom DPI and transparency:

    .. code-block:: python

        from alphapepttools.pl.figure import create_figure, save_figure

        fig, axm = create_figure(1, 2, figsize=(8, 4))
        # ... create plots ...

        save_figure(fig, "figure_transparent.pdf", "output/publication/", dpi=600, transparent=True)

    Notes
    -----
    The function automatically creates parent directories if they don't exist.
    The default DPI is taken from the global configuration settings to ensure
    consistent resolution across all saved figures.

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
