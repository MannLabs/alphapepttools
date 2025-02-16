# submodule to provide standardized colors and color maps

import colorsys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mpl_colors


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
    """Cycle through a palette to get n colors"""
    if n > len(palette):
        palette = palette * (n // len(palette) + 1)
    return palette[:n]


def _get_colors_from_cmap(
    cmap_name: str,
    num_colors: int,
) -> list:
    """Get a list of colors from a colormap"""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / (num_colors - 1)) for i in range(num_colors)]


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

    return picked_colors


def _base_binary_colorscale() -> list:
    """Base colorscale for binary data"""
    colors = _get_colors_from_cmap("BrBG", 10)

    # hand-selected colors
    color_indices = [2, 7]
    return [colors[i] for i in color_indices]


class BaseColors:
    """Base colors for AlphaTools plots"""

    _colorscale = _base_qualitative_colorscale()
    red = _colorscale[0]
    green = _colorscale[1]
    blue = _colorscale[2]
    orange = _colorscale[3]
    yellow = _colorscale[4]
    lightred = _colorscale[5]
    lightgreen = _colorscale[6]
    lightblue = _colorscale[7]
    lightorange = _colorscale[8]
    grey = mpl_colors.to_rgba("lightgrey")
    black = mpl_colors.to_rgba("black")
    white = mpl_colors.to_rgba("white")

    @classmethod
    def get(
        cls,
        color_name: str,
        lighten: float | None = None,
        alpha: float | None = None,
    ) -> tuple:
        """Get a default color by name, optionally lightened and/or with alpha"""
        try:
            color = getattr(cls, color_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown color name: {color_name}") from exc
        if lighten is not None:
            color = _lighten_color(color, lighten)
        if alpha is not None:
            color = tuple(list(color)[:3] + [alpha])
        return color


class BasePalettes:
    """Base color palettes for AlphaTools plots"""

    qualitative = _base_qualitative_colorscale()
    binary = _base_binary_colorscale()

    @classmethod
    def get(
        cls,
        palette_name: str,
        n: int | None = None,
    ) -> list:
        """Get a default color palette by name"""
        try:
            palette = getattr(cls, palette_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown palette name: {palette_name}") from exc

        # if n is greater than the length of the palette, loop through the palette
        if n is not None:
            palette = _cycle_palette(palette, n)

        return palette


class BaseColormaps:
    """Base colorscales for AlphaTools plots"""

    sequential = plt.get_cmap("mako")
    diverging = plt.get_cmap("vlag")

    @classmethod
    def get(
        cls,
        colorscale_name: str,
    ) -> list:
        """Get a default matplotlib.pyplot cmap by name"""
        try:
            colorscale = getattr(cls, colorscale_name)
        except AttributeError:
            try:
                colorscale = plt.get_cmap(colorscale_name)
            except ValueError as exc:
                raise ValueError(f"Unknown colorscale name: {colorscale_name}") from exc

        return colorscale


class MappedColormaps:
    """Mapped colorscales to numerical values in data

    Mapping a continuous colorscale to data (e.g. in a heatmap) requires
    that the values are normalized, that descending/ascending coloring is
    available, and that outliers can be compressed within a reasonable range.

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
        """Normalize data and transform it to colors"""
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
