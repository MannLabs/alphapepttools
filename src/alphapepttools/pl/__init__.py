from importlib import import_module as _import_module

from .colors import BaseColormaps, BaseColors, BasePalettes, MappedColormaps, get_color_mapping, show_rgba_color_list
from .figure import create_figure, label_axes, save_figure

__all__ = [
    "BaseColormaps",
    "BaseColors",
    "BasePalettes",
    "MappedColormaps",
    "PlotConfig",
    "Plots",
    "add_legend_to_axes",
    "add_legend_to_axes_from_patches",
    "add_lines",
    "barplot",
    "boxplot",
    "create_figure",
    "get_color_mapping",
    "histogram",
    "label_axes",
    "label_plot",
    "layered_plot",
    "make_scatter_config",
    "plot_pca",
    "plot_pca_loadings",
    "plot_pca_loadings_2d",
    "rank_median_plot",
    "save_figure",
    "scatter",
    "scree_plot",
    "show_rgba_color_list",
    "violinplot",
    "volcano",
]

_PLOT_EXPORTS = frozenset(name for name in __all__ if name not in globals())


def __getattr__(name: str) -> object:
    if name == "plots":
        return _import_module(".plots", __name__)

    if name not in _PLOT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(_import_module(".plots", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | {"plots"})
