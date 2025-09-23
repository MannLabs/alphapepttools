# at_colors.py:

# Defines the basic look and color layout of AlphaTools. This includes:

# - BaseColors (for getting individual RGBA colors in AlphaTools-style), defining colors based on
# - BasePalettes (for getting individual discrete palettes in AlphaTools-style), defined as a custom
#   rearranged version of the spectral colorscale for maximum separation when iterating.
# - BaseColormaps: Perceptually uniform colormaps (based on the https://github.com/callumrollo/cmcrameri.git)


import colorsys
import logging
from typing import ClassVar

import cmcrameri.cm as cmc
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mpl_colors
from matplotlib.colors import Colormap

from alphatools.pl import defaults

logger = logging.getLogger(__name__)

config = defaults.plot_settings.to_dict()


def show_rgba_color_list(colors: list) -> None:
    """Show a list of RGBA colors for quick inspection"""
    fig, ax = plt.subplots(figsize=(np.min([len(colors), 10]), 1))
    ax.imshow([colors], aspect="auto")
    ax.axis("off")
    plt.show()


def _lighten_color(
    color: tuple,
    factor: float = 0.5,
) -> tuple:
    """Lighten a color by a factor"""
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
    """Return a truncated version of a colormap."""
    new_colors = cmap(np.linspace(lowpoint, highpoint, n))
    return mcolors.LinearSegmentedColormap.from_list(f"{cmap.name}_trunc", new_colors)


def _cycle_palette(
    palette: list,
    n: int,
) -> list:
    """Cycle through a palette to get n colors. If n > len(palette), repeat the palette"""
    if n > len(palette):
        palette = palette * (n // len(palette) + 1)
    return palette[:n]


def _get_colors_from_cmap(
    cmap_name: str | mpl.colors.Colormap,
    values: int | np.ndarray,
) -> list | np.ndarray:
    """Retrieve colors from a colormap

    Parameters
    ----------
    cmap_name : str | mpl.colors.Colormap
        Name of the colormap or a Colormap instance. If a string, tries to retrieve the corresponding colormap from matplotlib.
    values : int | np.ndarray
        If int, retrieve that many evenly spaced colors from the colormap as a list of RGBA tuples.
        If np.ndarray, normalize the values to the range of the colormap and retrieve the corresponding colors in whatever shape the input array was. In
        the case of 2D input arrays, the output will be a mxnx4 array of RGBA tuples.

    Returns
    -------
    list[tuple[float, float, float, float]] | np.ndarray
        List of RGBA tuples if values is int, or an array of RGBA tuples with the same shape as values if values is np.ndarray.

    Examples
    --------
    >>> _get_colors_from_cmap("Spectral", np.ones((2, 2))).shape
    (2, 2, 4)
    >>> _get_colors_from_cmap("Spectral", 1)
    [(0.6196078431372549, 0.00392156862745098, 0.25882352941176473, 1.0)]

    """
    cmap = plt.get_cmap(cmap_name)

    if isinstance(values, int):
        return [tuple(color) for color in cmap(np.linspace(0, 1, values))]

    if not pd.api.types.is_numeric_dtype(values):
        raise TypeError("values must be an integer or a numeric numpy array")

    vmin, vmax = np.nanmin(values), np.nanmax(values)
    values = mpl_colors.Normalize(vmin=vmin, vmax=vmax)(values)
    return cmap(values)


def _base_qualitative_colorscale() -> list:
    """Base colorscale selected from matplotlib's 'Spectral' colorscale

    This colorscale aims for maximum separability when sequentially assigned to data:
    Start with four basic colors: red, green, blue, orange. If the colorscale is
    extended beyond 4 colors, bright yellow is introduced for maximum contrast.
    Beyond that, lightened versions of the 4 basic colors are used for a maximum of
    9 colors.

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
    """Base colorscale selected from a perceptually uniform colormap

    Sample along the lipari colorscale from cmcrameri to get 10 distinct colors

    """
    colors = _get_colors_from_cmap(cmc.batlow, 9)

    # interlace colors 1-5, 2-6, etc. to maximize color distance
    colors = list(zip(colors[:3], colors[3:6], colors[6:9], strict=False))
    return [color for pair in colors for color in pair]


def get_color_mapping(values: np.ndarray, palette: list[str | tuple] | mpl.colors.Colormap) -> dict:
    """Map categorical values to colors.

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

    """
    values = pd.unique(values.astype(str))

    # Set missing values aside for later addition with default NA-color
    na_values = np.array([v for v in values if v in config["na_identifiers"]])
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
    if na_values:
        result_dict.update(na_dict)

    return result_dict


def _base_binary_colorscale() -> list:
    """Base colorscale for binary data"""
    colors_left = _get_colors_from_cmap("cmc.batlow", 10)
    colors_right = _get_colors_from_cmap("cmc.devon", 10)

    # hand-selected colors
    return [_lighten_color(colors_left[7], 0.0), _lighten_color(colors_right[3], 0.0)]


class BaseColors:
    """Base colors for AlphaTools plots"""

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
    ) -> tuple:
        """Get a default color by name, optionally lightened and/or with alpha"""
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
    """Base color palettes for AlphaTools plots"""

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
        """Get a default color palette by name"""
        palette = None
        if palette_name in cls.default_palettes:
            palette = cls.default_palettes[palette_name]
        if palette is None:
            try:
                palette = _get_colors_from_cmap(palette_name, values=n)
            except ValueError as exc:
                raise ValueError(f"Unknown palette name: {palette_name}") from exc

        return _cycle_palette(palette, n)


# TODO: Fix so a defined number of colors can be retrieved with .get, as well as the whole colormap
class BaseColormaps:
    """Base colormaps for AlphaTools plots"""

    # Use perceptually uniform color palettes to avoid visual distortion (Crameri, F. (2018a), Scientific colour maps. Zenodo. http://doi.org/10.5281/zenodo.1243862)
    default_colormaps: ClassVar[dict] = {
        "sequential": cmc.devon,
        "diverging": cmc.managua_r,
        "sequential_r": cmc.devon_r,
        "diverging_r": cmc.managua,
        "sequential_clipped": clip_colormap(cmc.devon, lowpoint=0, highpoint=0.8),
        "sequential_r_clipped": clip_colormap(cmc.devon_r, lowpoint=0.2, highpoint=1),
        "magma_clipped": clip_colormap(plt.get_cmap("magma"), lowpoint=0, highpoint=0.8),
    }

    @classmethod
    def get(
        cls,
        colormap_name: str,
        n: int | None = None,
    ) -> Colormap | list:
        """Get a default matplotlib.pyplot cmap by name

        Optionally, specify the number of colors to be retrieved from the colormap. defaults
        are set for colormap_name = "sequential" and "diverging". Otherwise, colormap_name is
        passed to plt.get_cmap().

        Parameters
        ----------
        colormap_name : str
            Name of the colormap to be used
        n : int, optional
            Number of colors to be retrieved from the colormap. If None, the full colormap is returned.

        Returns
        -------
        Colormap | list
            Matplotlib Colormap object or list of colors

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
    """Mapped colorscales to numerical values in data

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

    Attributes
    ----------
    cmap : Colormap
        Matplotlib Colormap object to be used for mapping data to colors
    vmin : float
        Minimum value of the data used for normalization
    vmax : float
        Maximum value of the data used for normalization
    color_normalizer : mpl.colors.Normalize
        Normalization instance used to map data to colors based on vmin and vmax
    scalar_mappable : mpl.cm.ScalarMappable
        MappedColormaps(color_map, percentile).scalar_mappable is a mpl.cm.ScalarMappable instance of the current
        colormap and normalized data values which can be used in colorbars.

    """

    def __init__(
        self,
        cmap: str,
        percentile: tuple[float, float] | None = None,
    ):
        self.cmap = BaseColormaps.get(cmap)
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

        Parameters
        ----------
        data : np.ndarray
            Data to be transformed into colors. Based on this data, the colormap will be normalized.
        """
        data = np.asarray(data).copy()

        if self.percentile is not None:
            self.vmin = np.nanpercentile(data, self.percentile[0])
            self.vmax = np.nanpercentile(data, self.percentile[1])
        else:
            self.vmin = np.nanmin(data)
            self.vmax = np.nanmax(data)

        data = np.clip(data, self.vmin, self.vmax)

        rgba = _get_colors_from_cmap(self.cmap, data)

        if as_hex:
            return np.apply_along_axis(mpl_colors.to_hex, -1, rgba, keep_alpha=True)
        return rgba

    @property
    def scalar_mappable(self) -> mpl.cm.ScalarMappable:
        """Return a ScalarMappable for use in colorbars"""
        if self.vmin is None or self.vmax is None:
            raise ValueError("fit_transform must be called before accessing scalar_mappable")
        sm = plt.cm.ScalarMappable(norm=mpl_colors.Normalize(vmin=self.vmin, vmax=self.vmax), cmap=self.cmap)
        sm.set_array([])
        return sm


def invert_color(
    color: tuple | str,
) -> tuple | str:
    """Invert a color

    Parameters
    ----------
    color : tuple | str
        Color to be inverted. Can be an RGBA tuple or a color name.

    Returns
    -------
    Tuple | str
        Inverted color as RGBA tuple or hex string.
    """
    if isinstance(color, str):
        color = mpl_colors.to_rgba(color)

    inverted_color = (*tuple(1 - np.array(color[:3])), color[3])  # Preserve alpha channel if present

    return mpl_colors.to_hex(inverted_color, keep_alpha=True) if isinstance(color, str) else inverted_color
