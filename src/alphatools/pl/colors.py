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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mpl_colors
from matplotlib.colors import Colormap


def show_rgba_color_list(colors: list) -> None:
    """Show a list of RGBA colors for quick inspection"""
    fig, ax = plt.subplots(figsize=(10, 1))
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


def _cycle_palette(
    palette: list,
    n: int,
) -> list:
    """Cycle through a palette to get n colors. If n > len(palette), repeat the palette"""
    if n > len(palette):
        palette = palette * (n // len(palette) + 1)
    return palette[:n]


def _get_colors_from_cmap(
    cmap_name: str,
    num_colors: int,
) -> list:
    """Use a matplotlib colormap to get a list of colors"""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i) for i in np.linspace(0, 1, num_colors)]


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


def get_color_mapping(values: np.ndarray, palette: list[str | tuple] | mpl.colors.Colormap) -> dict:
    """Map values to colors

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
    values = np.unique(values)

    if isinstance(palette, list):
        _palette = _cycle_palette(palette, n=len(values))
    elif isinstance(palette, mpl.colors.Colormap):
        _palette = _get_colors_from_cmap(palette, num_colors=len(values))
    else:
        raise TypeError("palette must be a list of colors or a matplotlib colormap")

    return dict(zip(values, _palette, strict=True))


def _base_binary_colorscale() -> list:
    """Base colorscale for binary data"""
    colors = _get_colors_from_cmap("BrBG", 10)

    # hand-selected colors
    color_indices = [2, 7]
    return [_lighten_color(colors[i], 0.1) for i in color_indices]


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
        "qualitative": _base_qualitative_colorscale(),
        "binary": _base_binary_colorscale(),
    }

    @classmethod
    def get(
        cls,
        palette_name: str,
        n: int = 10,
    ) -> list:
        """Get a default color palette by name"""
        if palette_name in cls.default_palettes:
            palette = cls.default_palettes[palette_name]
        if palette is None:
            try:
                palette = _get_colors_from_cmap(palette_name, num_colors=n)
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
        if colormap_name in cls.default_colormaps:
            colormap = cls.default_colormaps[colormap_name]
        else:
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

    """

    def __init__(
        self,
        cmap: str,
        percentile: tuple[float, float] | None = None,
    ):
        self.cmap = BaseColormaps.get(cmap)
        self.percentile = percentile

    def fit_transform(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """Normalize data and transform it to colors

        Parameters
        ----------
        data : np.ndarray
            Data to be transformed into colors. Based on this data, the colormap will be normalized.
        """
        data = np.array(data.copy())

        if self.percentile is not None:
            self.vmin = np.percentile(data, self.percentile[0])
            self.vmax = np.percentile(data, self.percentile[1])
        else:
            self.vmin = np.min(data)
            self.vmax = np.max(data)

        data = np.clip(data, self.vmin, self.vmax)

        self.color_normalizer = mpl_colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        normalized_data = self.color_normalizer(data)

        return [self.cmap(d) for d in normalized_data]
