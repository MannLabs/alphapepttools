# at_colors.py:

# Defines the basic look and color layout of alphapepttools. This includes:

# - BaseColors (for getting individual RGBA colors in alphapepttools-style), defining colors based on
# - BasePalettes (for getting individual discrete palettes in alphapepttools-style), defined as a custom
#   rearranged version of the spectral colorscale for maximum separation when iterating.
# - BaseColormaps: Perceptually uniform colormaps (based on the https://github.com/callumrollo/cmcrameri.git)


import colorsys
import logging
from typing import Any, ClassVar, cast

import cmcrameri.cm as cmc
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mpl_colors
from matplotlib.colors import Colormap

from alphapepttools.pl import defaults

logger = logging.getLogger(__name__)

config = defaults.plot_settings.to_dict()


def show_rgba_color_list(colors: list) -> None:
    """Display a horizontal bar of RGBA colors for visual inspection

    Creates a matplotlib figure showing the provided colors as a horizontal bar,
    useful for quickly visualizing color palettes or verifying color schemes.
    The figure width is automatically adjusted based on the number of colors
    (up to a maximum of 10 units).

    Parameters
    ----------
    colors : list
        List of RGBA color tuples, where each tuple contains (red, green, blue, alpha)
        values in the range [0, 1].

    Returns
    -------
    None

    Examples
    --------
    Visualize a custom color palette:

    .. code-block:: python

        colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
        show_rgba_color_list(colors)

    Inspect colors from a colormap:

    .. code-block:: python

        from alphapepttools.pl.colors import BaseColormaps

        cmap_colors = BaseColormaps.get("sequential", n=5)
        show_rgba_color_list(cmap_colors)

    """
    fig, ax = plt.subplots(figsize=(np.min([len(colors), 10]), 1))
    ax.imshow([colors], aspect="auto")
    ax.axis("off")
    plt.show()


def _lighten_color(
    color: tuple,
    factor: float = 0.5,
) -> tuple:
    """Adjust the lightness of a color by a given factor

    Converts the color to HLS (Hue, Lightness, Saturation) space, adjusts the
    lightness component, and converts back to RGB(A). Positive factors lighten
    the color, negative factors darken it. The alpha channel is preserved if present.

    Parameters
    ----------
    color : tuple
        RGB or RGBA color tuple with values in the range [0, 1].
    factor : float, default=0.5
        Lightness adjustment factor. Positive values lighten the color by moving
        toward white, negative values darken by moving toward black. The adjustment
        is applied to the interval between the current lightness and the target
        (1 for positive, 0 for negative).

    Returns
    -------
    tuple
        Adjusted RGB or RGBA color tuple with values in the range [0, 1].

    Examples
    --------
    Lighten a red color:

    .. code-block:: python

        red = (1.0, 0.0, 0.0, 1.0)
        light_red = _lighten_color(red, factor=0.3)

    Darken a blue color:

    .. code-block:: python

        blue = (0.0, 0.0, 1.0)
        dark_blue = _lighten_color(blue, factor=-0.5)

    """
    if len(color) == 4:  # noqa: PLR2004
        # dealing with RGBA
        alpha = color[-1]
        color = color[:3]
    else:
        alpha = None

    r, g, b = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # apply factor to lightness interval between current l to 1 and add to current l
    l = min(1, l + factor * (1 - l))

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    color = [r, g, b]

    if alpha is not None:
        color += [alpha]

    return tuple(color)


def clip_colormap(
    cmap: mpl.colors.Colormap,
    lowpoint: float = 0.0,
    highpoint: float = 1.0,
    n: int = 256,
) -> mcolors.LinearSegmentedColormap:
    """Create a truncated version of a colormap by clipping to a specified range

    Extracts a subset of colors from an existing colormap between the specified
    low and high points, creating a new LinearSegmentedColormap. This is useful
    for avoiding extreme colors (e.g., very dark or very light regions) that may
    have poor visibility or contrast.

    Parameters
    ----------
    cmap : mpl.colors.Colormap
        The source colormap to be truncated.
    lowpoint : float, default=0.0
        Lower bound of the colormap range to extract, in the range [0, 1].
    highpoint : float, default=1.0
        Upper bound of the colormap range to extract, in the range [0, 1].
    n : int, default=256
        Number of discrete color levels in the resulting colormap.

    Returns
    -------
    mcolors.LinearSegmentedColormap
        New colormap containing only colors between lowpoint and highpoint.

    Examples
    --------
    Clip the devon colormap to avoid very dark colors:

    .. code-block:: python

        import cmcrameri.cm as cmc

        clipped = clip_colormap(cmc.devon, lowpoint=0, highpoint=0.8)

    Create a symmetric diverging colormap from the middle range:

    .. code-block:: python

        import matplotlib.pyplot as plt

        viridis = plt.get_cmap("viridis")
        mid_range = clip_colormap(viridis, lowpoint=0.2, highpoint=0.8)

    """
    new_colors = cmap(np.linspace(lowpoint, highpoint, n))
    return mcolors.LinearSegmentedColormap.from_list(f"{cmap.name}_trunc", new_colors)


def _cycle_palette(
    palette: list,
    n: int,
) -> list:
    """Extend a color palette to n colors by cycling through the palette

    If the requested number of colors exceeds the palette length, the palette
    is repeated as many times as necessary. This ensures that all requested
    colors are assigned, though colors may be reused when n > len(palette).

    Parameters
    ----------
    palette : list
        List of colors (as RGBA tuples, hex strings, or color names).
    n : int
        Number of colors to return.

    Returns
    -------
    list
        List of n colors, cycling through the original palette if necessary.

    Examples
    --------
    Extend a 3-color palette to 5 colors:

    .. code-block:: python

        palette = ["red", "green", "blue"]
        extended = _cycle_palette(palette, n=5)  # ["red", "green", "blue", "red", "green"]

    Request fewer colors than available:

    .. code-block:: python

        palette = ["red", "green", "blue", "yellow"]
        subset = _cycle_palette(palette, n=2)  # ["red", "green"]

    """
    if n > len(palette):
        palette = palette * (n // len(palette) + 1)
    return palette[:n]


def _get_colors_from_cmap(
    cmap_name: str | mpl.colors.Colormap,
    values: int | np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> list | np.ndarray:
    """Retrieve colors from a colormap for discrete or continuous data

    This function supports two modes of operation: (1) extracting a fixed number
    of evenly-spaced colors for categorical data, or (2) mapping continuous numeric
    values to colors by normalizing the data range to the colormap range.

    Parameters
    ----------
    cmap_name : str | mpl.colors.Colormap
        Name of the colormap or a Colormap instance. If a string, tries to retrieve
        the corresponding colormap from matplotlib.
    values : int | np.ndarray
        If int, retrieve that many evenly spaced colors from the colormap as a list
        of RGBA tuples. If np.ndarray, normalize the values to the range of the
        colormap and retrieve the corresponding colors in whatever shape the input
        array was. In the case of 2D input arrays, the output will be a mxnx4 array
        of RGBA tuples.
    vmin : float, optional
        Minimum value for normalization when values is an array. If None, uses the minimum of the array.
    vmax : float, optional
        Maximum value for normalization when values is an array. If None, uses the maximum of the array.

    Returns
    -------
    list[tuple[float, float, float, float]] | np.ndarray
        List of RGBA tuples if values is int, or an array of RGBA tuples with the
        same shape as values if values is np.ndarray.

    Examples
    --------
    Extract 5 evenly spaced colors from a colormap:

    .. code-block:: python

        colors = _get_colors_from_cmap("viridis", 5)
        # Returns list of 5 RGBA tuples

    Map continuous data to colors preserving array shape:

    .. code-block:: python

        import numpy as np

        data = np.random.rand(3, 4)
        colors = _get_colors_from_cmap("Spectral", data)
        # Returns (3, 4, 4) array where last dimension is RGBA

    Single color from a colormap:

    .. code-block:: python

        color = _get_colors_from_cmap("Spectral", 1)
        # [(0.619..., 0.003..., 0.258..., 1.0)]

    """
    cmap = plt.get_cmap(cmap_name)

    if isinstance(values, int):
        return [tuple(color) for color in cmap(np.linspace(0, 1, values))]

    if not pd.api.types.is_numeric_dtype(values):
        raise TypeError("values must be an integer or a numeric numpy array")

    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)

    values = mpl_colors.Normalize(vmin=vmin, vmax=vmax)(values)
    return cmap(values)


def _base_qualitative_colorscale() -> list:
    """Generate base qualitative colorscale with maximum color separability

    This colorscale is derived from matplotlib's 'Spectral' colormap and aims for
    maximum separability when sequentially assigned to categorical data. The palette
    starts with four basic colors: red, green, blue, orange. If extended beyond 4
    colors, bright yellow is introduced for maximum contrast. Beyond that, lightened
    versions of the 4 basic colors are used for a maximum of 9 distinct colors.

    Returns
    -------
    list
        List of 9 RGBA tuples optimized for categorical data visualization.

    Examples
    --------
    Retrieve the base qualitative colorscale:

    .. code-block:: python

        colors = _base_qualitative_colorscale()
        # Returns 9 carefully selected RGBA colors
        show_rgba_color_list(colors)

    """
    colors = _get_colors_from_cmap("Spectral", 12)

    # hand-selected colors
    color_indices = [0, 9, 10, 2, 5, 1, 8, 10, 3]
    picked_colors = [colors[i] for i in color_indices]

    # handle special case: light blue
    picked_colors[-2] = _lighten_color(picked_colors[-2], 0.3)

    # handle special case: light red
    picked_colors[-4] = _lighten_color(picked_colors[-4], 0.2)

    # handle special case: darken yellow
    picked_colors[4] = _lighten_color(picked_colors[4], -1)

    return picked_colors


def _perceptually_uniform_qualitative_colorscale() -> list:
    """Generate perceptually uniform qualitative colorscale for categorical data

    This colorscale is sampled from the cmcrameri batlow colormap. Since this colormap
    is meant for qualitative data where differentiating different levels is paramount,
    the colors are interlaced and lightness-adjusted to maximize perceptual distance
    between sequential colors when assigned to categorical data.

    Returns
    -------
    list
        List of 9 RGBA tuples with perceptually uniform color distribution.

    Examples
    --------
    Retrieve the perceptually uniform qualitative colorscale:

    .. code-block:: python

        colors = _perceptually_uniform_qualitative_colorscale()
        # Returns 9 perceptually uniform RGBA colors
        show_rgba_color_list(colors)

    """
    colors = _get_colors_from_cmap(cmc.cmaps["batlow"], 9)

    # interlace colors 1-5, 2-6, etc. to maximize color distance
    # Custom hue for maximum contrast between levels
    return [
        colors[0],
        _lighten_color(colors[3], 0.1),
        colors[6],
        _lighten_color(colors[1], 0.3),
        _lighten_color(colors[4], 0.3),
        _lighten_color(colors[7], -0.3),
        _lighten_color(colors[2], 0.3),
        _lighten_color(colors[5], 0.0),
        _lighten_color(colors[8], 0.0),
    ]


def get_color_mapping(values: np.ndarray, palette: list[str | tuple] | mpl.colors.Colormap) -> dict:
    """Map categorical values to colors

    Maps unique values in `values` to colors from `palette`. If `palette` is a list of colors,
    colors are assigned in a cycling manner, which can lead to non-unique assignments if there
    are more unique values than colors in the palette. If `palette` is a colormap, colors are
    assigned uniquely based on the number of unique values. Missing values (as defined in
    `config["na_identifiers"]`) are assigned the default NA color (`config["na_color"]`).

    Parameters
    ----------
    values : np.ndarray
        Values to map to colors
    palette : list[str | tuple] | mpl.colors.Colormap
        If palette is a list, it is used as a discrete colormap with cycling colors (potential for non-unique assignment).
        If palette is a colormap, it is used as a continuous colormap (guaranteed unique assignment).

    Returns
    -------
    dict
        Dictionary mapping values to colors

    Examples
    --------
    Map categorical levels to a color palette:

    .. code-block:: python

        import numpy as np

        levels = np.array(["A", "B", "C", "A", "B"])
        palette = ["red", "green", "blue"]
        color_dict = get_color_mapping(levels, palette)

    Map using a matplotlib colormap:

    .. code-block:: python

        import matplotlib.pyplot as plt

        levels = np.array(["low", "medium", "high"])
        cmap = plt.get_cmap("viridis")
        color_dict = get_color_mapping(levels, cmap)

    """
    values = pd.unique(values.astype(str))

    # Set missing values aside for later addition with default NA-color
    na_values = [v for v in values if v in config["na_identifiers"]]
    na_dict = dict(zip(na_values, [config["na_color"]] * len(na_values), strict=True))

    # Continue with non-missing values only
    values = np.array([v for v in values if v not in config["na_identifiers"]])

    # Ensure predictable color mapping
    values = np.sort(values)

    # Map color levels to a palette or a colormap
    if isinstance(palette, list):
        _palette = _cycle_palette(palette, n=len(values))
    elif isinstance(palette, mpl.colors.Colormap):
        # Use the color mapping with an integer number of values, i.e. get equally spaced colors from the colormap
        _palette = _get_colors_from_cmap(palette, values=len(values))
    else:
        raise TypeError("palette must be a list of colors or a matplotlib colormap")

    result_dict = dict(zip(values, _palette, strict=True))

    # Add missing values to the mapping with the default na color
    if len(na_values) > 0:
        result_dict.update(na_dict)

    return result_dict


def _base_binary_colorscale() -> list:
    """Generate base colorscale for binary data with high contrast

    Selects two colors from complementary cmcrameri colormaps (batlow and devon)
    to maximize contrast and aesthetics for binary data visualization.

    Returns
    -------
    list
        List of 2 RGBA tuples optimized for binary data visualization.

    Examples
    --------
    Retrieve the binary colorscale:

    .. code-block:: python

        colors = _base_binary_colorscale()
        # Returns 2 high-contrast RGBA colors
        show_rgba_color_list(colors)

    """
    colors_left = _get_colors_from_cmap("cmc.batlow", 10)
    colors_right = _get_colors_from_cmap("cmc.devon", 10)

    # hand-selected colors along the respective palettes to obtain good contrast and aesthetics
    return [colors_left[7], colors_right[3]]


class BaseColors:
    """Default color palette for alphapepttools plots

    Provides access to a curated set of default colors optimized for data
    visualization. Colors are derived from the base qualitative colorscale
    with additional utility colors (grey, black, white).

    """

    _colorscale = _base_qualitative_colorscale()
    default_colors: ClassVar[dict] = {
        "red": _colorscale[0],
        "green": _colorscale[1],
        "blue": _colorscale[2],
        "orange": _colorscale[3],
        "yellow": _colorscale[4],
        "lightred": _colorscale[5],
        "lightgreen": _colorscale[6],
        "lightblue": _colorscale[7],
        "lightorange": _colorscale[8],
        "grey": mpl_colors.to_rgba("lightgrey"),
        "black": mpl_colors.to_rgba("black"),
        "white": mpl_colors.to_rgba("white"),
    }

    @classmethod
    def get(
        cls,
        color_name: str | tuple,
        lighten: float | None = None,
        alpha: float | None = None,
    ) -> tuple | str:
        """Retrieve a color by name with optional lightness and alpha adjustments

        Retrieves colors from the default color palette, matplotlib named colors,
        or passes through RGBA tuples. Optionally adjusts lightness and alpha
        transparency.

        Parameters
        ----------
        color_name : str | tuple
            Name of color from default_colors, matplotlib color name, or RGBA tuple.
        lighten : float, optional
            Lightness adjustment factor. Positive values lighten, negative darken.
        alpha : float, optional
            Alpha transparency value in range [0, 1].

        Returns
        -------
        tuple
            RGBA color tuple with values in range [0, 1].

        Examples
        --------
        Get a default color:

        .. code-block:: python

            from alphapepttools.pl.colors import BaseColors

            red_color = BaseColors.get("red")
            # Returns RGBA tuple for red

        Get a lightened color:

        .. code-block:: python

            light_blue = BaseColors.get("blue", lighten=0.3)
            # Returns lightened version of blue

        Get a color with custom alpha:

        .. code-block:: python

            transparent_red = BaseColors.get("red", alpha=0.5)
            # Returns red with 50% transparency

        Use matplotlib color names:

        .. code-block:: python

            color = BaseColors.get("coral")
            # Retrieves matplotlib's coral color

        """
        # First, avoid trying to map RGBA tuples to colors
        if isinstance(color_name, tuple):
            color = color_name
        # Second, check if the color name is available in the defaults
        elif color_name in cls.default_colors:
            color = cls.default_colors[color_name]
        # Third, try to get the color from matplotlib
        else:
            try:
                color = mpl_colors.to_rgba(color_name)
            except ValueError:
                logging.warning(f"Unknown color name: {color_name}, cannot parse to RGBA or change lightness/alpha")
                return color_name

        if lighten is not None:
            color = _lighten_color(color, lighten)

        if alpha is not None:
            color = tuple(list(color)[:3] + [alpha])

        return color


class BasePalettes:
    """Discrete color palettes for alphapepttools plots

    Provides access to curated discrete color palettes optimized for categorical
    data visualization. Palettes can be retrieved by name and will cycle to
    provide the requested number of colors.

    Attributes
    ----------
    default_palettes : dict
        Dictionary of available palettes:
        'qualitative_spectral' - Spectral-based palette with maximum separability
        'qualitative' - Perceptually uniform palette from cmcrameri
        'binary' - Two-color palette for binary data

    """

    default_palettes: ClassVar[dict] = {
        "qualitative_spectral": _base_qualitative_colorscale(),
        "qualitative": _perceptually_uniform_qualitative_colorscale(),
        "binary": _base_binary_colorscale(),
    }

    @classmethod
    def get(
        cls,
        palette_name: str,
        n: int = 9,
    ) -> list:
        """Retrieve a color palette by name with specified number of colors

        Retrieves palettes from the default palette collection or matplotlib
        colormaps. Automatically cycles the palette to provide the requested
        number of colors.

        Parameters
        ----------
        palette_name : str
            Name of palette from default_palettes or matplotlib colormap name.
        n : int, default=9
            Number of colors to return. Palette will cycle if n exceeds palette length.

        Returns
        -------
        list
            List of n RGBA color tuples.

        Examples
        --------
        Get a default palette:

        .. code-block:: python

            from alphapepttools.pl.colors import BasePalettes

            colors = BasePalettes.get("qualitative", n=5)
            # Returns 5 colors from the qualitative palette

        Get a palette from matplotlib:

        .. code-block:: python

            colors = BasePalettes.get("viridis", n=10)
            # Retrieves 10 colors from the viridis colormap

        Get binary colors for two groups:

        .. code-block:: python

            colors = BasePalettes.get("binary", n=2)
            # Returns 2 high-contrast colors

        """
        palette = None
        if palette_name in cls.default_palettes:
            palette = cls.default_palettes[palette_name]
        if palette is None:
            try:
                palette = _get_colors_from_cmap(palette_name, values=n)
            except ValueError as exc:
                raise ValueError(f"Unknown palette name: {palette_name}") from exc

        # both branches above assign a list (`_get_colors_from_cmap` with an int returns a list)
        return _cycle_palette(cast("list", palette), n)


# TODO: Fix so a defined number of colors can be retrieved with .get, as well as the whole colormap
class BaseColormaps:
    """Continuous colormaps for alphapepttools plots

    Provides access to perceptually uniform colormaps optimized for continuous
    data visualization. Uses cmcrameri colormaps to avoid visual distortion
    (Crameri, F. (2018a), Scientific colour maps. Zenodo.
    http://doi.org/10.5281/zenodo.1243862).

    """

    # Use perceptually uniform color palettes to avoid visual distortion (Crameri, F. (2018a), Scientific colour maps. Zenodo. http://doi.org/10.5281/zenodo.1243862)
    default_colormaps: ClassVar[dict] = {
        "sequential": cmc.cmaps["devon"],
        "diverging": cmc.cmaps["managua_r"],
        "sequential_r": cmc.cmaps["devon_r"],
        "diverging_r": cmc.cmaps["managua"],
        "sequential_clipped": clip_colormap(cmc.cmaps["devon"], lowpoint=0, highpoint=0.8),
        "sequential_r_clipped": clip_colormap(cmc.cmaps["devon_r"], lowpoint=0.2, highpoint=1),
        "magma_clipped": clip_colormap(plt.get_cmap("magma"), lowpoint=0, highpoint=0.8),
    }

    @classmethod
    def get(
        cls,
        colormap_name: str,
        n: int | None = None,
    ) -> Colormap | list:
        """Retrieve a colormap by name or sample discrete colors from it

        Retrieves colormaps from the default colormap collection or matplotlib.
        Can return either the full continuous colormap or a discrete set of
        n evenly-spaced colors sampled from it.

        Parameters
        ----------
        colormap_name : str
            Name of colormap from default_colormaps or matplotlib colormap name.
        n : int, optional
            Number of discrete colors to sample from the colormap. If None, returns
            the full continuous colormap object.

        Returns
        -------
        Colormap | list
            If n is None, returns matplotlib Colormap object. If n is specified,
            returns list of n RGBA color tuples sampled evenly from the colormap.

        Examples
        --------
        Get a full colormap for continuous data:

        .. code-block:: python

            from alphapepttools.pl.colors import BaseColormaps

            cmap = BaseColormaps.get("sequential")
            # Returns continuous colormap object for use with imshow, contourf, etc.

        Sample discrete colors from a colormap:

        .. code-block:: python

            colors = BaseColormaps.get("sequential", n=5)
            # Returns 5 evenly-spaced RGBA colors from sequential colormap

        Use a diverging colormap:

        .. code-block:: python

            cmap = BaseColormaps.get("diverging")
            # Returns diverging colormap centered at midpoint

        Get clipped colormap to avoid extreme colors:

        .. code-block:: python

            cmap = BaseColormaps.get("sequential_clipped")
            # Returns sequential colormap with darkest 20% removed

        """
        colormap = None
        if colormap_name in cls.default_colormaps:
            colormap = cls.default_colormaps[colormap_name]

        if colormap is None:
            try:
                colormap = plt.get_cmap(colormap_name)
            except ValueError as exc:
                raise ValueError(f"Unknown colormap name: {colormap_name}") from exc

        if n is not None:
            return [colormap(i) for i in np.linspace(0, 1, n)]

        return colormap


class MappedColormaps:
    """Percentile-based colormap normalization for outlier-robust visualization

    Mapping a continuous colorscale to data can suffer from compression due to outliers.
    For example, if 90 % of values lie between 0 and 1 but 10 % of values are between 1 and 100,
    these extreme values will cause all other values to be compressed into a small range of colors.
    By applying normalization to a certain percentile range of the data, the colormap can be adjusted
    accordingly. Values outside the percentile will receive the same color as the minimum or maximum,
    respectively.

    Parameters
    ----------
    cmap : str
        Name of the colormap to be used

    percentile : tuple[float, float], optional
        Percentile range to be used for normalization. If None, the full range of data is used.
        For example, (5, 95) will map colors between the 5th and 95th percentile.

    """

    def __init__(
        self,
        cmap: str,
        percentile: tuple[float, float] | None = None,
    ):
        """Initialize MappedColormaps with a colormap and optional percentile range for normalization

        Parameters
        ----------
        cmap : str
            Name of the colormap to be used for mapping data to colors. Can be a name from BaseColormaps or any matplotlib colormap name.
        percentile : tuple[float, float], optional
            Percentile range to be used for normalization. If None, the full range of data is used. For example, (5, 95) will map colors between the 5th and 95th percentile of the data.

        Returns
        -------
        None

        Example

        """
        # `get` without `n` always returns a continuous Colormap, never the discrete-color list
        self.cmap = cast("Colormap", BaseColormaps.get(cmap))
        self.percentile = percentile
        self.vmin = None
        self.vmax = None

    def fit_transform(
        self,
        data: np.ndarray,
        *,
        as_hex: bool = False,
    ) -> np.ndarray:
        """Normalize data and transform it to colors

        Convenience wrapper around :meth:`fit` followed by :meth:`transform`: the
        normalization bounds are derived from the data (the full range, or the
        configured ``percentile`` range) and the data is then mapped to colors.
        Values outside the normalization range are clamped to the colormap's end
        colors. Use :meth:`fit` and :meth:`transform` separately to reuse one color
        scale across several arrays with differing ranges.

        Parameters
        ----------
        data : np.ndarray
            Data to be transformed into colors. Based on this data, the colormap
            will be normalized. Can be any shape; colors are returned in the same shape.
        as_hex : bool, default=False
            If True, return hex color strings. If False, return RGBA tuples.

        Returns
        -------
        np.ndarray
            Array of colors with the same shape as input data. If as_hex=False,
            the last dimension will be 4 (RGBA). If as_hex=True, returns array
            of hex strings.

        Examples
        --------
        Transform 1D data to colors:

        .. code-block:: python

            import numpy as np
            from alphapepttools.pl.colors import MappedColormaps

            data = np.array([1, 2, 3, 100])  # 100 is outlier
            mapper = MappedColormaps(cmap="sequential", percentile=(5, 95))
            colors = mapper.fit_transform(data)
            # Returns (4, 4) array of RGBA colors

        Transform 2D heatmap data:

        .. code-block:: python

            data = np.random.randn(10, 10)
            mapper = MappedColormaps(cmap="diverging")
            colors = mapper.fit_transform(data)
            # Returns (10, 10, 4) array of RGBA colors

        Get hex colors for plotting:

        .. code-block:: python

            data = np.array([1, 2, 3, 4, 5])
            mapper = MappedColormaps(cmap="sequential")
            hex_colors = mapper.fit_transform(data, as_hex=True)
            # Returns array of hex strings like ['#1a2b3c', ...]

        """
        return self.fit(data).transform(data, as_hex=as_hex)

    def fit(
        self,
        data: np.ndarray | None = None,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> "MappedColormaps":
        """Determine and store the normalization bounds without producing colors

        Bounds are derived from ``data`` (the full range, or the configured
        ``percentile`` range), unless explicit ``vmin``/``vmax`` are given, which
        override on a per-bound basis.

        Parameters
        ----------
        data : np.ndarray, optional
            Data from which to derive the bounds. Required unless both ``vmin`` and
            ``vmax`` are provided. NaNs are ignored.
        vmin : float, optional
            Explicit lower bound, overriding the data-derived minimum.
        vmax : float, optional
            Explicit upper bound, overriding the data-derived maximum.

        Returns
        -------
        MappedColormaps
            The fitted instance (``self``).

        Raises
        ------
        ValueError
            If a bound cannot be determined from ``data`` or explicit values.

        Examples
        --------
        Pin the colormap to fixed bounds without any data:

        .. code-block:: python

            mapper = MappedColormaps(cmap="cmc.vik")
            mapper.fit(vmin=0, vmax=1)

        """
        if data is not None:
            arr = np.asarray(data)
            if self.percentile is not None:
                d_min = np.nanpercentile(arr, self.percentile[0])
                d_max = np.nanpercentile(arr, self.percentile[1])
            else:
                d_min, d_max = np.nanmin(arr), np.nanmax(arr)
        else:
            d_min = d_max = None

        self.vmin = vmin if vmin is not None else d_min
        self.vmax = vmax if vmax is not None else d_max

        if self.vmin is None or self.vmax is None:
            raise ValueError("fit() requires `data`, or both `vmin` and `vmax`, to set the bounds.")
        return self

    def transform(
        self,
        data: np.ndarray,
        *,
        as_hex: bool = False,
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        """Map data to colors using the previously fitted normalization bounds

        Uses the ``vmin``/``vmax`` set by :meth:`fit` or :meth:`fit_transform`
        without re-deriving them from ``data``. Values outside ``[vmin, vmax]`` are
        clamped to the colormap's end colors.

        Parameters
        ----------
        data : np.ndarray
            Data to be transformed into colors. Can be any shape; colors are
            returned in the same shape.
        as_hex : bool, default=False
            If True, return hex color strings. If False, return RGBA tuples.

        Returns
        -------
        np.ndarray
            Array of colors with the same shape as input data. If as_hex=False,
            the last dimension will be 4 (RGBA). If as_hex=True, returns array
            of hex strings.

        Raises
        ------
        ValueError
            If called before :meth:`fit` or :meth:`fit_transform`.

        """
        if self.vmin is None or self.vmax is None:
            raise ValueError("Call fit() or fit_transform() before transform().")

        rgba = _get_colors_from_cmap(self.cmap, np.asarray(data), vmin=self.vmin, vmax=self.vmax)
        if as_hex:
            return np.apply_along_axis(mpl_colors.to_hex, -1, rgba, keep_alpha=True)
        return np.asarray(rgba)

    @property
    def scalar_mappable(self) -> mpl.cm.ScalarMappable:
        """Return a ScalarMappable for use in colorbars

        This property provides a ScalarMappable instance that can be used to create colorbars consistent with the normalization applied in fit_transform.
        It uses the same vmin, vmax, and colormap, ensuring that the colorbar accurately reflects the mapping of data values to colors.

        Returns
        -------
        mpl.cm.ScalarMappable
            ScalarMappable instance with the same colormap and normalization as used in fit_transform.

        """
        if self.vmin is None or self.vmax is None:
            raise ValueError("fit_transform must be called before accessing scalar_mappable")
        sm = plt.cm.ScalarMappable(norm=mpl_colors.Normalize(vmin=self.vmin, vmax=self.vmax), cmap=self.cmap)
        sm.set_array([])
        return sm


def invert_color(
    color: tuple | str,
) -> tuple | str:
    """Invert a color by subtracting RGB values from 1

    Inverts the RGB components of a color while preserving the alpha channel.
    Useful for creating contrasting colors or generating color opposites for
    visualization purposes.

    Parameters
    ----------
    color : tuple | str
        Color to be inverted. Can be an RGBA tuple with values in [0, 1] or a
        color name/hex string.

    Returns
    -------
    tuple | str
        Inverted color in the same format as input. If input is a tuple, returns
        RGBA tuple. If input is a string, returns hex string.

    Examples
    --------
    Invert an RGBA tuple:

    .. code-block:: python

        from alphapepttools.pl.colors import invert_color

        red = (1.0, 0.0, 0.0, 1.0)
        inverted = invert_color(red)
        # Returns (0.0, 1.0, 1.0, 1.0) - cyan

    Invert a color name:

    .. code-block:: python

        inverted = invert_color("blue")
        # Returns hex string for inverted blue (yellow)

    Invert a hex color:

    .. code-block:: python

        inverted = invert_color("#FF5733")
        # Returns hex string for inverted color

    """
    if isinstance(color, str):
        color = mpl_colors.to_rgba(color)

    inverted_color = (*tuple(1 - np.array(color[:3])), color[3])  # Preserve alpha channel if present

    return mpl_colors.to_hex(inverted_color, keep_alpha=True) if isinstance(color, str) else inverted_color
