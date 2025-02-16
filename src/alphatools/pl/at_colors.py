# submodule to provide standardized colors and color maps

import colorsys

import matplotlib.pyplot as plt
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


def _base_colorscale() -> list:
    """Base colorscale selected from matplotlib's 'Spectral' colorscale

    This colorscale aims for maximum separability when sequentially assigned to data:
    Start with four basic colors: red, green, blue, orange. If the colorscale is
    extended beyond 4 colors, bright yellow is introduced for maximum contrast.
    Beyond that, lightened versions of the 4 basic colors are used for a maximum of
    9 colors.

    """
    cmap = plt.get_cmap("Spectral")
    num_colors = 11
    colors = [cmap(i / (num_colors)) for i in range(num_colors)]

    # hand-selected colors
    color_indices = [0, 9, 10, 2, 5, 1, 8, 10, 3]
    picked_colors = [colors[i] for i in color_indices]

    # handle special case: light blue
    picked_colors[-2] = _lighten_color(picked_colors[-2], 0.3)

    # handle special case: light red
    picked_colors[-4] = _lighten_color(picked_colors[-4], 0.2)

    return picked_colors


class BaseColors:
    """Base colors for AlphaTools plots"""

    def __init__(self):
        self._colorscale = _base_colorscale()
        self.red = self._colorscale[0]
        self.green = self._colorscale[1]
        self.blue = self._colorscale[2]
        self.orange = self._colorscale[3]
        self.yellow = self._colorscale[4]
        self.lightred = self._colorscale[5]
        self.lightgreen = self._colorscale[6]
        self.lightblue = self._colorscale[7]
        self.lightorange = self._colorscale[8]
        self.grey = mpl_colors.to_rgba("lightgrey")
        self.black = mpl_colors.to_rgba("black")
        self.white = mpl_colors.to_rgba("white")
