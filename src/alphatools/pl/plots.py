# plots.py

# Main plotting submodule with a private method for generating legends, and a
# Plots class containing class methods to generate plots. The proposed
# layout for plotting functions is such that they accept AnnData objects and dataframes.
# When columns to plot are specified for an AnnData object, the _adata_column_to_array()
# function first tries to find the column in the var_names (i.e. the columns of the actual
# data), and then in the obs.columns (for example, when plotting a numeric value from X and
# coloring it by a metadata column from obs, see 03_basic_workflow.ipynb).

import logging
from collections import Counter
from collections.abc import Callable
from typing import Any

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from alphatools.pl import defaults
from alphatools.pl.colors import BaseColors, BasePalettes, _get_colors_from_cmap, get_color_mapping
from alphatools.pl.figure import create_figure, label_axes
from alphatools.pp.data import data_column_to_array

# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = defaults.plot_settings.to_dict()


def add_lines(
    ax: plt.Axes,
    intercepts: float | list[float | int],
    linetype: str = "vline",
    color: str = "black",
    linestyle: str = "--",
    linewidth: float = 1,
    line_kwargs: dict | None = None,
) -> None:
    """Add a vertical or horizontal line to a matplotlib axes object

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to add the line to.
    linetype : str
        Type of line to add, either 'vline' or 'hline'.
    intercepts : float | list[float | int]
        Intercepts of the line(s) to add.
    color : str, optional
        Color of the line(s), by default "black".
    linestyle : str, optional
        Linestyle of the line(s), by default "--".
    linewidth : float, optional
        Linewidth of the line(s), by default 1.
    line_kwargs : dict, optional
        Additional keyword arguments for the line function, by default None. Will be overridden by color, linestyle, and linewidth arguments.

    Returns
    -------
    None

    """
    if linetype not in ["vline", "hline"]:
        raise ValueError("linetype must be 'vline' or 'hline'")
    line_func = ax.axvline if linetype == "vline" else ax.axhline

    if not isinstance(intercepts, (list | float | int)):
        raise TypeError("intercepts must be a float, int, or list of floats/ints")

    # handle intercepts and vertical/horizontal lines
    if isinstance(intercepts, float | int):
        intercepts = [intercepts]

    # handle clashes between keyword arguments and line_kwargs
    line_kwargs = line_kwargs or {}
    color = line_kwargs.pop("color", color)
    linestyle = line_kwargs.pop("linestyle", linestyle)
    linewidth = line_kwargs.pop("linewidth", linewidth)

    # add lines to ax
    for intercept in intercepts:
        line_func(
            intercept,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            **line_kwargs,
        )


def make_legend_patches(
    color_dict: dict[str, str | tuple],
) -> list[mpl.patches.Patch]:
    """Create legend patches for a matplotlib legend from a value-to-color mapping

    This is a helper function for the add_legend function.
    Matplotlib legends display labelled patches with a defined color. This function
    takes a dictionary of values and colors and returns a list of named patches.

    Parameters
    ----------
    color_dict : dict[str, str | tuple]
        Dictionary of values and colors.

    Returns
    -------
    list[mpl.patches.Patch]
        List of named patches.
    """
    patches = []
    for value, color in color_dict.items():
        patches.append(
            Patch(
                label=value,
                facecolor=color,
                edgecolor=BaseColors.get("black"),
                linewidth=config["linewidths"]["medium"],
            )
        )

    return patches


def add_legend_to_axes_from_patches(
    ax: plt.Axes,
    patches: list[mpl.patches.Patch],
    **kwargs,
) -> None:
    """Make a legend and directly add it to a matplotlib axes object.

    Expects a list of named patches.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to add the legend to.
    patches : list[mpl.patches.Patch]
        List of patches to use for the legend.

    Returns
    -------
    None

    """
    # create new legend
    if "fontsize" not in kwargs:
        kwargs["fontsize"] = config["legend"]["font_size"]

    _legend = ax.legend(handles=patches, **kwargs)

    # Resize legend title based on config legend title_size
    _legend.set_title(_legend.get_title().get_text(), prop={"size": config["legend"]["title_size"]})


def add_legend_to_axes(
    ax: plt.Axes,
    levels: list[str] | dict[str, str | tuple] | None = None,
    legend: str | mpl.legend.Legend | None = "auto",
    palette: list[str | tuple] | None = None,
    **legend_kwargs,
) -> None:
    """Add a legend to an axis object.

    Handle legend creation in three ways:
    1.: 'levels' is a dictionary of levels and colors, in which case these levels and colors are used directly.
    2.: 'levels' is a list of levels, in which case a color palette is used to assign colors to levels. A custom
    palette can be provided, otherwise a default palette is used.
    3.: 'legend' is a matplotlib legend object, which overrides all other options and is added directly to the axes.
    This defaults to 'auto', which directs to the first two cases.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to add the legend to.
    levels : list[str] | dict[str, str | tuple] | None
        List of levels to use for the legend. Duplicates are removed. Colors from the palette are assigned to unique values from this list,
        but no particular color-binding is enforced. If this is a dictionary, the legend contains exactly the labels (keys) and colors (values) provided.
    legend : str | mpl.legend.Legend | None
        Legend to add to the plot. If "auto", a legend is created based on levels. If a Legend object, it is added directly to the axes. By default "auto".
    palette : list[str | tuple] | None
        List of colors to use for the legend. If None, a default palette will be used. By default None. Only relevant when levels is a list, i.e. when matching
        of values to colors happens automatically.
    legend_kwargs : dict, optional
        Additional keyword arguments for the legend, by default {}. This can include 'fontsize', 'title', etc. These kwargs are not enforced if a matplotlib legend object
        is passed as the `legend` parameter.

    """
    if isinstance(legend, mpl.legend.Legend):
        ax.add_artist(legend)
        return
    if legend == "auto":
        if isinstance(levels, dict):
            patches = make_legend_patches(levels)
            add_legend_to_axes_from_patches(ax, patches, **legend_kwargs)
        elif isinstance(levels, list):
            levels = np.unique(levels)
            if palette is None:
                palette = BasePalettes.get("qualitative")
                if len(levels) > len(palette):
                    palette = BasePalettes.get("sequential")
            color_dict = get_color_mapping(levels, palette)
            patches = make_legend_patches(color_dict)
            add_legend_to_axes_from_patches(ax, patches, **legend_kwargs)
        else:
            logging.warning("No valid 'levels' parameter provided. Skipping legend creation.")
    else:
        logging.warning("No valid 'legend' parameter provided. Skipping legend creation.")


def _drop_nans_from_plot_arrays(
    x_values: np.ndarray,
    y_values: np.ndarray,
    labels: np.ndarray | list[str],
) -> tuple:
    # Missing x or y values are breaking and should be dropped
    keep_mask = ~np.logical_or(pd.isna(x_values), pd.isna(y_values))

    return x_values[keep_mask], y_values[keep_mask], labels[keep_mask]


def _assign_nearest_anchor_position_to_values(
    values: np.ndarray,
    anchors: list[int | float] | np.ndarray | None,
) -> list:
    if anchors is None:
        return values

    # x-values are binned to the anchor positions
    anchored_values = []

    for val in values:
        anchor_diffs = [abs(anchor - val) for anchor in anchors]
        anchored_values.append(anchors[np.argmin(anchor_diffs)])

    return anchored_values


def label_plot(
    ax: plt.Axes,
    x_values: list | np.ndarray,
    y_values: list | np.ndarray,
    labels: list[str] | np.ndarray,
    x_anchors: list[int | float] | np.ndarray | None = None,
    label_kwargs: dict | None = None,
    line_kwargs: dict | None = None,
    label_parser: Callable | None = None,
    y_display_start: float = 1,
    y_padding_factor: float = 3,
) -> None:
    """Add labels to a 2D axes object

    Add labels to a plot based on x and y coordinates. The labels are either placed near the datapoint
    using the automatic dodging function from adjust_text or anchored to the left or right of the plot,
    where labels below the splitpoint are anchored to the left and labels above the splitpoint are anchored
    to the right.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to add the labels to.
    x_values : list | np.ndarray
        x-coordinates of the labels.
    y_values : list | np.ndarray
        y-coordinates of the labels.
    labels : list[str] | np.ndarray
        Labels to add to the plot.
    x_anchors : list[int | float] | np.ndarray | None, optional
        x-coordinates of the anchors to use for the labels. If None, labels are placed at the x-coordinates of the data points. By default None.
    label_kwargs : dict | None, optional
        Additional keyword arguments for the label text, by default None.
    line_kwargs : dict | None, optional
        Additional keyword arguments for the line connecting the label to the data point, by default None.
    label_parser : Callable | None, optional
        Function to parse the labels, by default None. This is useful to convert
        labels from a computation-context to presentation context, e.g. a column
        like upregulated_proteins could be shown as "Upregulated Proteins" in the plot.
    y_display_start : float, optional
        Starting point for the y-coordinates of the labels, by default 1. This is used to determine the spacing between labels.
        The y-coordinates of the labels are adjusted to be evenly spaced between the min and max y-coordinates at that anchor.
        This is useful for avoiding label overlap.
    y_padding_factor: float, optional
        Factor to increase or decrease how far apart labels are spread in the y-direction when stacked into a column over x-anchors

    """
    label_kwargs = {"fontsize": config["font_sizes"]["medium"], **(label_kwargs or {})}
    line_kwargs = {"color": BaseColors.get("black"), "linewidth": config["linewidths"]["medium"], **(line_kwargs or {})}
    label_parser = label_parser or (lambda x: x)

    if not len(x_values) == len(y_values) == len(labels):
        raise ValueError("x_values, y_values, and labels must have the same length")

    # convert to numpy arrays for consistency & remove any nans
    x_values, y_values, labels = _drop_nans_from_plot_arrays(np.array(x_values), np.array(y_values), np.array(labels))

    # determine label positions based on optional x_anchors
    if x_anchors is not None:
        # x-values are binned to the anchor positions
        label_x_values = _assign_nearest_anchor_position_to_values(x_values, x_anchors)

        # y-values should be distributed evenly between the min and max y-values at that anchor
        label_spacing_display = config["font_sizes"]["medium"] * y_padding_factor

        # Translate label spacing from display coordinates to axes coordinates, since the same spacing should appear regardless of y-values
        transform = ax.transData.inverted()
        _, y_spacing_in_data_coords = transform.transform((0, label_spacing_display)) - transform.transform((0, 0))

        # get a consistent starting point for y values with respect to the actual display window
        _, upper_bound_in_data_coords = transform.transform((0, ax.get_window_extent().height * y_display_start))

        # Iterate over all unique x_anchors and assign y-values in data coordinates to the respective labels
        label_y_values = []
        for anchor in np.unique(label_x_values):
            current_label_y_values = np.sort(y_values[np.array(label_x_values) == anchor])
            label_y_values.extend(
                [upper_bound_in_data_coords - y_spacing_in_data_coords * i for i in range(len(current_label_y_values))]
            )
    else:
        label_x_values = x_values
        label_y_values = y_values

    # generate lines from data values to label positions
    lines = []
    for label, x, y, label_x, label_y in zip(labels, x_values, y_values, label_x_values, label_y_values, strict=False):
        lines.append(((x, label_x), (y, label_y), label))

    for line in lines:
        ax.plot(line[0], line[1], **line_kwargs)
        if x_anchors is not None:
            alignment = "right" if line[0][0] > line[0][1] else "left"
            label_kwargs["ha"] = alignment
        ax.text(line[0][1], line[1][1], label_parser(line[2]), **label_kwargs)


def _validate_pca_plot_input(
    data: ad.AnnData,
    pca_embeddings_layer_name: str,
    pca_variance_layer_name: str,
    pc_x: int,
    pc_y: int,
    dim_space: str,
) -> None:
    """
    Validates the AnnData object for PCA-related data and dimensions.

    Parameters
    ----------
    data:
        AnnData object to be validated.
    pca_embeddings_layer_name:
        Name of the PCA layer to be checked.
    pca_variance_layer_name:
        Name of the column for explained variance to be checked.
    pc_x:
        First PCA dimension to be validated (1-indexed, i.e. the first PC is 1, not 0).
    pc_y:
        Second PCA dimension to be validated (1-indexed, i.e. the first PC is 1, not 0).
    dim_space:
        The dimension space used in PCA. Can be either "obs" or "var".
    """
    if not isinstance(data, ad.AnnData):
        raise TypeError("data must be an AnnData object")

    if dim_space not in ["obs", "var"]:
        raise ValueError(f"dim_space must be either 'obs' or 'var', got {dim_space}")

    # Determine which attribute to check based on dim_space
    pca_coors_attr = "obsm" if dim_space == "obs" else "varm"

    # Check if the PCA embeddings layer exists in the correct attribute
    if pca_embeddings_layer_name not in getattr(data, pca_coors_attr):
        available_layers = list(getattr(data, pca_coors_attr).keys())
        raise ValueError(
            f"PCA embeddings layer '{pca_embeddings_layer_name}' not found in data.{pca_coors_attr}. "
            f"Found layers: {available_layers}"
        )

    # Check if the variance layer exists in uns
    if pca_variance_layer_name not in data.uns:
        raise ValueError(
            f"PCA metadata layer '{pca_variance_layer_name}' not found in AnnData object. "
            f"Found layers: {list(data.uns.keys())}"
        )

    # Check PC dimensions
    n_pcs = getattr(data, pca_coors_attr)[pca_embeddings_layer_name].shape[1]
    if not (1 <= pc_x <= n_pcs) or not (1 <= pc_y <= n_pcs):
        raise ValueError(f"pc_x and pc_y must be between 1 and {n_pcs} (inclusive). Got pc_x={pc_x}, pc_y={pc_y}.")


def _validate_scree_plot_input(
    data: ad.AnnData,
    pca_variance_layer_name: str,
    n_pcs: int,
) -> None:
    """
    Validate inputs for scree plot of the PCA dimension.

    Parameters
    ----------
    data : anndata.AnnData
        The AnnData object containing PCA results.
    pca_variance_layer_name : str
        The name of the PCA layer (used to construct the embedding key as `data.uns[pca_name]`).
    n_pcs : int
        The number of principal components requested for plotting.

    """
    if not isinstance(data, ad.AnnData):
        raise TypeError("data must be an AnnData object")

    if pca_variance_layer_name not in data.uns:
        raise ValueError(
            f"PCA metadata layer '{pca_variance_layer_name}' not found in AnnData object. "
            f"Found layers: {list(data.uns.keys())}"
        )

    n_pcs_avail = len(data.uns[pca_variance_layer_name]["variance_ratio"])
    if n_pcs > n_pcs_avail:
        logging.warning(
            f"Requested {n_pcs} PCs, but only {n_pcs_avail} PCs are available. Plotting only the available PCs."
        )


def _validate_pca_loadings_plot_inputs(
    data: ad.AnnData, loadings_name: str, dim: int, dim2: int | None, nfeatures: int, dim_space: str
) -> None:
    """
    Validate inputs for accessing PCA feature loadings from an AnnData object.

    Parameters
    ----------
    data : anndata.AnnData
        The AnnData object containing PCA loadings data.
    loadings_name : str
        The key that stores PCA feature loadings (e.g., "PCs").
    dim: int
        The principal component index (1-based) to extract loadings for.
    dim2 : int | None
        The second principal component index (1-based) to extract loadings for, if applicable.
    nfeatures : int
        The number of top features to consider for the given component.
    dim_space : str
        The dimension space used in PCA. Can be either "obs" or "var".
    """
    if not isinstance(data, ad.AnnData):
        raise TypeError("data must be an AnnData object")

    if dim_space not in ["obs", "var"]:
        raise ValueError(f"dim_space must be either 'obs' or 'var', got {dim_space}")

    # Determine which attribute to check based on dim_space
    loadings_attr = "varm" if dim_space == "obs" else "obsm"

    # Check if the loadings layer exists in the correct attribute
    if loadings_name not in getattr(data, loadings_attr):
        available_layers = list(getattr(data, loadings_attr).keys())
        raise ValueError(
            f"PCA feature loadings layer '{loadings_name}' not found in data.{loadings_attr}. "
            f"Found layers: {available_layers}"
        )

    # Check PC dimensions
    n_pcs = getattr(data, loadings_attr)[loadings_name].shape[1]
    if not (1 <= dim <= n_pcs):
        raise ValueError(f"PC must be between 1 and {n_pcs} (inclusive). Got dim={dim}.")
    if dim2 is not None and not (1 <= dim2 <= n_pcs):
        raise ValueError(f"second PC must be between 1 and {n_pcs} (inclusive). Got pc_y={dim2}.")

    # Check number of features
    n_features = getattr(data, loadings_attr)[loadings_name].shape[0]
    if not (1 <= nfeatures <= n_features):
        raise ValueError(
            f"Number of features must be between 1 and {n_features} (inclusive). Got nfeatures={nfeatures}."
        )


def _prepare_loading_df_to_plot(
    data: ad.AnnData, loadings_name: str, pc_x: int, pc_y: int, nfeatures: int, dim_space: str
) -> pd.DataFrame:
    """
    Prepare a DataFrame with PCA feature loadings for plotting.

    This function extracts the loadings of two specified principal components (PCs) from
    an AnnData object, filters features that contributed to the PCA (non-zero loadings),
    and flags the top nfeatures for each selected PC dimension.

    Parameters
    ----------
    data : anndata.AnnData
        The AnnData object containing PCA results.
    loadings_name : str
        The key where PCA loadings are stored.
    pc_x : int
        The first principal component index (1-based) to extract loadings for.
    pc_y : int
        The second principal component index (1-based) to extract loadings for.
    nfeatures : int
        Number of top features per PC to highlight based on absolute loadings.
    dim_space : str
        The dimension space used in PCA. Can be either "obs" or "var".

    Returns
    -------
    pd.DataFrame
        DataFrame containing loadings for the selected PCs, feature names, boolean columns
        indicating if a feature was used in PCA and whether it is among the top features in either dimension.
    """
    _validate_pca_loadings_plot_inputs(
        data=data, loadings_name=loadings_name, dim=pc_x, dim2=pc_y, nfeatures=nfeatures, dim_space=dim_space
    )

    dim1_z = pc_x - 1  # convert to 0-based index
    dim2_z = pc_y - 1  # convert to 0-based index

    # Determine which attribute to use based on dim_space
    loadings_attr = "varm" if dim_space == "obs" else "obsm"
    orig_loadings = getattr(data, loadings_attr)[loadings_name]

    loadings = pd.DataFrame(
        {
            "dim1_loadings": orig_loadings[:, dim1_z],
            "dim2_loadings": orig_loadings[:, dim2_z],
        }
    )

    # Add feature names based on dim_space
    if dim_space == "obs":
        loadings["feature"] = data.var_names
    else:  # dim_space == "var"
        loadings["feature"] = data.obs_names

    # get only features that were used in the PCA (e.g., those that are part of the core proteome)
    # these would be features with all-NaN loadings in all PC dimensions
    non_nan_mask = ~np.isnan(orig_loadings).all(axis=1)
    loadings = loadings[non_nan_mask]

    # add the top N features for each dimension
    loadings["abs_dim1"] = loadings["dim1_loadings"].abs()
    loadings["abs_dim2"] = loadings["dim2_loadings"].abs()

    loadings["is_top"] = False
    loadings.loc[loadings.nlargest(nfeatures, "abs_dim1").index, "is_top"] = True
    loadings.loc[loadings.nlargest(nfeatures, "abs_dim2").index, "is_top"] = True

    return loadings


def _array_to_str(
    array: np.ndarray | pd.Series,
) -> np.ndarray:
    """Map a numpy array to string values, while replacing NaNs with default nan-filler string."""
    string_array = np.array(array, dtype=object)

    if config["na_default"] in string_array:
        logger.warning(
            f"The default NaN replacement string '{config['na_default']}' is present in the data. Consider choosing a different value to avoid overwriting."
        )

    string_array[pd.isna(string_array)] = config["na_default"]  # replace NaNs with default string
    return string_array.astype(str)  # ensure all values are strings


def _dict_keys_to_str(
    dictionary: dict,
) -> dict[str, Any]:
    """Convert the keys of a dictionary to strings."""
    return {str(k): v for k, v in dictionary.items()}


class Plots:
    """Class for creating figures with matplotlib

    Basic configuration for matplotlib plots is loaded from a YAML file
    and set to generate consistent plots.

    """

    def __init__(
        self,
        config: dict = defaults.plot_settings.to_dict(),
    ):
        self.config = config

    @classmethod
    def histogram(
        cls,
        data: pd.DataFrame | ad.AnnData,
        value_column: str,
        color_map_column: str | None = None,
        bins: int = 10,
        ax: plt.Axes | None = None,
        color: str = "blue",
        palette: list[tuple] | None = None,
        color_dict: dict[str, str | tuple] | None = None,
        legend: str | mpl.legend.Legend | None = None,
        hist_kwargs: dict | None = None,
        legend_kwargs: dict | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """Plot a histogram from a DataFrame or AnnData object

        Parameters
        ----------
        data : pd.DataFrame | ad.AnnData
            Data to plot, must contain the value_column and optionally the color_column.
        value_column : str
            Column in data to plot as histogram. Must contain numeric data.
        color_map_column : str, optional
            Column in data to use for color encoding. These values are mapped to the palette or the color_dict (see below). Its values cannot contain NaNs, therefore color_map_column is coerced to string and missing values replaced by a default filler string. Overrides color parameter. By default None.
        bins : int, optional
            Number of bins to use for the histogram. By default 10.
        color : str, optional
            Color to use for the histogram. By default "blue".
        ax : plt.Axes, optional
            Matplotlib axes object to plot on, if None a new figure is created. By default None.
        palette : list[tuple], optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
        color_dict: dict[str, str | tuple], optional
            Supercedes palette, a dictionary mapping levels to colors. By default None. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
        hist_kwargs : dict, optional
            Additional keyword arguments for the matplotlib hist function. By default None.
        legend_kwargs : dict, optional
            Additional keyword arguments for the matplotlib legend function. By default None.
        xlim : tuple[float, float], optional
            Limits for the x-axis. By default None.
        ylim : tuple[float, float], optional
            Limits for the y-axis. By default None.

        Returns
        -------
        None

        """
        hist_kwargs = hist_kwargs or {}
        legend_kwargs = legend_kwargs or {}

        if ax is None:
            _, ax = create_figure(1, 1)

        values = data_column_to_array(data, value_column)

        if color_map_column is None:
            color = BaseColors.get(color)
            ax.hist(values, bins=bins, color=color, **hist_kwargs)
        else:
            color_levels = _array_to_str(data_column_to_array(data, color_map_column))
            color_dict = _dict_keys_to_str(
                color_dict or get_color_mapping(color_levels, palette or BasePalettes.get("qualitative"))
            )

            for level in set(color_levels) - set(color_dict):
                color_dict[level] = BaseColors.get("grey")

            for level, level_color in color_dict.items():
                ax.hist(
                    values[color_levels == level],
                    bins=bins,
                    color=level_color,
                    **hist_kwargs,
                )

            if legend is not None:
                add_legend_to_axes(
                    ax=ax,
                    levels=color_dict,
                    legend=legend,
                    **legend_kwargs,
                )

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    @classmethod
    def scatter(
        cls,
        data: pd.DataFrame | ad.AnnData,
        x_column: str,
        y_column: str,
        color: str | None = None,
        color_map_column: str | None = None,
        color_column: str | None = None,
        ax: plt.Axes | None = None,
        palette: list[str | tuple] | None = None,
        color_dict: dict[str, str | tuple] | None = None,
        legend: str | mpl.legend.Legend | None = None,
        scatter_kwargs: dict | None = None,
        legend_kwargs: dict | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """Plot a scatterplot from a DataFrame or AnnData object

        Coloring works in three ways, with the following order of precedence: 1. color_column, 2. color_map_column, 3. color.
        If a color_column is provided, its values are interpreted directly as colors, i.e. they have to be something matplotlib
        can understand (e.g. RGBA, hex, etc.). If a color_map_column is provided, its values are mapped to colors in combination
        with palette or color_dict (see color mapping logic below). If neither color_column nor color_map_column is provided, the
        color parameter is used to color all points the same (defaults to blue).

        Color mapping logic
        -------------------
        - color_map_column is non-numeric:
            - If color_dict is not None: Use color_dict to assign levels of color_map_column to colors (unmapped levels default to grey).
            - If color_dict is None, and palette is not None: Use palette to automatically assign colors to each level.
            - If color_dict is None and palette is None: Use a repeating default palette to assign colors to each level.
        - color_map_column is numeric:
            - If palette is a matplotlib colormap: Numerically map values to colors using the colormap. This means that e.g. 1 and 3 will be closer in color than 1 and 10.
            - If palette is not a matplotlib colormap: Treat numeric values as categorical and color as described above.

        - Examples:
            - color_column="my_colors": Points colored by values in "my_colors" column (must contain valid colors)
            - color_map_column="cell_type": Categorical mapping of cell types to colors
            - color_map_column="expression", palette=plt.cm.viridis: Continuous gradient based on expression values


        Parameters
        ----------
        data : pd.DataFrame | ad.AnnData
            Data to plot, must contain the x_column and y_column and optionally the color_column or color_map_column.
        x_column : str
            Column in data to plot on the x-axis. Must contain numeric data.
        y_column : str
            Column in data to plot on the y-axis. Must contain numeric data.
        color : str, optional
            Color to use for the scatterplot. By default "blue".
        color_map_column : str, optional
            Column in data to use for color encoding. These values are mapped to the palette or the color_dict (see below). Its values cannot contain NaNs, therefore color_map_column is coerced to string and missing values replaced by a default filler string. Overrides color parameter. By default None.
        color_column : str, optional
            Column in data to plot the colors. This must contain actual color values (RGBA, hex, etc.). Overrides color and color_map_column parameters. By default None.
        ax : plt.Axes, optional
            Matplotlib axes object to plot on, if None a new figure is created. By default None.
        palette : list[str | tuple] | matplotlib.colors.Colormap, optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
        color_dict: dict[str, str | tuple], optional
            Supercedes palette, a dictionary mapping levels to colors. By default None. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.
        legend_kwargs : dict, optional
            Additional keyword arguments for the matplotlib legend function. By default None.
        xlim : tuple[float, float], optional
            Limits for the x-axis. By default None.
        ylim : tuple[float, float], optional
            Limits for the y-axis. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}
        legend_kwargs = legend_kwargs or {}
        DEFAULT_GROUP = "data"
        DEFAULT_COLOR = BaseColors.get("blue")

        if ax is None:
            _, axm = create_figure()
            ax = axm.next()

        # Directly use colors from the color_column
        if color_column is not None:
            color_values = data_column_to_array(data, color_column)
        # Map values from the color_map_column to colors
        elif color_map_column is not None:
            color_map_column_array = data_column_to_array(data, color_map_column)

            if pd.api.types.is_numeric_dtype(color_map_column_array) and isinstance(palette, plt.Colormap):
                color_values = _get_colors_from_cmap(
                    cmap_name=palette,
                    values=color_map_column_array,
                )
            # if color_map_column is not numeric
            else:
                color_map_column_array = _array_to_str(data_column_to_array(data, color_map_column))
                color_dict = _dict_keys_to_str(
                    color_dict
                    or get_color_mapping(
                        values=color_map_column_array, palette=palette or BasePalettes.get("qualitative")
                    )
                )
                for level in set(color_map_column_array) - set(color_dict):
                    color_dict[level] = BaseColors.get("grey")
                color_values = np.array([color_dict[level] for level in color_map_column_array], dtype=object)
        else:
            color_dict = {DEFAULT_GROUP: color or DEFAULT_COLOR}
            color_values = np.array([color_dict[DEFAULT_GROUP]] * len(data))

        # Handle ordering of plotting arrays by string: order by the frequency of the color column
        counts = Counter([str(cv) for cv in color_values])
        order = np.argsort([counts[str(cv)] for cv in color_values])[::-1]
        x_values = data_column_to_array(data, x_column)[order]
        y_values = data_column_to_array(data, y_column)[order]
        color_values = np.array(color_values)[order]

        ax.scatter(
            x=x_values,
            y=y_values,
            c=color_values,
            **scatter_kwargs,
        )

        if legend is not None and color_dict is not None:
            add_legend_to_axes(
                ax=ax,
                levels=color_dict,
                legend=legend,
                **legend_kwargs,
            )

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    @classmethod
    def rank_median_plot(
        cls,
        data: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        layer: str = "X",
        color: str = "blue",
        color_map_column: str | None = None,
        color_column: str | None = None,
        palette: list[str | tuple] | None = None,
        color_dict: dict[str, str | tuple] | None = None,
        legend: str | mpl.legend.Legend | None = None,
        scatter_kwargs: dict | None = None,
    ) -> None:
        """Plot the ranked protein median intensities across all samples using the scatter method

        Parameters
        ----------
        data : ad.AnnData
            AnnData to plot.
        ax : plt.Axes
            Matplotlib axes object to plot on, add labels and logscale the y-axis.
        layer : str
            The AnnData layer to calculate the median value (intensities) across sample. Default is "X"
        color : str, optional
            Color to use for the scatterplot. By default "blue".
        color_map_column : str, optional
            Column in data to use for color encoding. These values are mapped to the palette or the color_dict (see below). Its values cannot contain NaNs, therefore color_map_column is coerced to string and missing values replaced by a default filler string. Overrides color parameter. By default None.
        color_column : str, optional
            Column in data to plot the colors. This must contain actual color values (RGBA, hex, etc.). Overrides color and color_map_column parameters. By default None.
        palette : list[str | tuple], optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
        color_dict: dict[str, str | tuple], optional
            A dictionary mapping levels to colors. By default None. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        if ax is None:
            _, axm = create_figure()
            ax = axm.next()

        if layer != "X" and layer not in data.layers:
            raise ValueError(f"Layer {layer} not found in AnnData object")

        # Use AnnData's dataframe extraction to get the values + annotations
        values = (data.to_df() if layer == "X" else data.to_df(layer=layer)) if isinstance(data, ad.AnnData) else data

        # compute medians and sort
        medians = values.median(axis=0).sort_values(ascending=False).to_frame(name="median")

        # Retain information about the proteins
        medians = medians.join(data.var) if isinstance(data, ad.AnnData) else medians
        medians["rank"] = np.arange(1, len(medians) + 1)

        # call the Plots.scatter method to create the rank plot
        cls.scatter(
            data=medians,
            x_column="rank",
            y_column="median",
            color=color,
            color_column=color_column,
            color_map_column=color_map_column,
            legend=legend,
            palette=palette,
            color_dict=color_dict,
            ax=ax,
            scatter_kwargs=scatter_kwargs,
        )

        # Adjust scale and labelling
        ax.set_yscale("log")

        label_axes(
            ax,
            xlabel="Rank",
            ylabel="Median Intensity",
        )

    @classmethod
    def plot_pca(
        cls,
        data: ad.AnnData,
        ax: plt.Axes,
        pc_x: int = 1,
        pc_y: int = 2,
        dim_space: str = "obs",
        embbedings_name: str | None = None,
        label: bool = False,  # noqa: FBT001, FBT002
        label_column: str | None = None,
        color: str = "blue",
        color_map_column: str | None = None,
        color_column: str | None = None,
        palette: list[str | tuple] | None = None,
        color_dict: dict[str, str | tuple] | None = None,
        legend: str | mpl.legend.Legend | None = None,
        scatter_kwargs: dict | None = None,
    ) -> None:
        """Plot the PCs of a PCA analysis using the scatter method

        Parameters
        ----------
        adata : ad.AnnData
            AnnData to plot.
        ax : plt.Axes
            Matplotlib axes object to plot on.
        pc_x : int
            The PC principal component index to plot on the x axis, by default 1. Corresponds to the principal component order, the first principal is 1 (1-indexed, i.e. the first PC is 1, not 0).
        pc_y : int
            The principal component index to plot on the y axis, by default 2. Corresponds to the principal component order, the first principal is 1 (1-indexed, i.e. the first PC is 1, not 0).
        dim_space : str, optional
            The dimension space used in PCA. Can be either "obs" (default) for sample projection or "var" for feature projection. By default "obs".
        embbedings_name : str | None, optional
            The custom embeddings name used in PCA (given as input for `pca` function in `embbedings_name` ). If None, uses default naming convention. By default None.
        label: bool,
            Whether to add labels to the points in the scatter plot. by default False.
        label_column: str | None = None,
            Column in data.obs to use for labeling the points. If None, and label is True, data.obs.index labels are added. By default None.
        color : str, optional
            Color to use for the scatterplot. By default "blue".
        color_map_column : str, optional
            Column in data to use for color encoding. These values are mapped to the palette or the color_dict (see below). Its values cannot contain NaNs, therefore color_map_column is coerced to string and missing values replaced by a default filler string. Overrides color parameter. By default None.
        color_column : str, optional
            Column in data to plot the colors. This must contain actual color values (RGBA, hex, etc.). Overrides color and color_map_column parameters. By default None.
        palette : list[str | tuple], optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
        color_dict: dict[str, str | tuple], optional
            Supercedes palette, a dictionary mapping levels to colors. By default None. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        # Generate the correct key names based on dim_space and embbedings_name
        pca_coors_key = f"X_pca_{dim_space}" if embbedings_name is None else embbedings_name
        variance_key = f"variance_pca_{dim_space}" if embbedings_name is None else embbedings_name

        # Determine which attribute to use for coordinates based on dim_space
        pca_coors_attr = "obsm" if dim_space == "obs" else "varm"

        # Input checks
        _validate_pca_plot_input(data, pca_coors_key, variance_key, pc_x, pc_y, dim_space)

        # create the dataframe for plotting
        dim1_z = pc_x - 1  # to account for 0 indexing
        dim2_z = pc_y - 1  # to account for 0 indexing

        # Get PCA coordinates from the correct attribute
        pca_coordinates = getattr(data, pca_coors_attr)[pca_coors_key]
        values = pd.DataFrame(pca_coordinates[:, [dim1_z, dim2_z]], columns=["dim1", "dim2"])

        # get the explained variance ratio for the dimensions
        var_dim1 = data.uns[variance_key]["variance_ratio"][dim1_z]
        var_dim1 = round(var_dim1 * 100, 2)
        var_dim2 = data.uns[variance_key]["variance_ratio"][dim2_z]
        var_dim2 = round(var_dim2 * 100, 2)

        # add color column
        if color_map_column is not None:
            color_values = data_column_to_array(data, color_map_column)
            values[color_map_column] = color_values

        cls.scatter(
            data=values,
            x_column="dim1",
            y_column="dim2",
            color=color,
            color_column=color_column,
            color_map_column=color_map_column,
            legend=legend,
            palette=palette,
            color_dict=color_dict,
            ax=ax,
            scatter_kwargs=scatter_kwargs,
        )

        # add labels if requested
        if label:
            # For labeling, we need to consider the appropriate observation space
            if dim_space == "obs":
                labels = data.obs.index if label_column is None else data_column_to_array(data, label_column)
            else:  # dim_space == "var"
                labels = data.var.index if label_column is None else data_column_to_array(data, label_column)

            label_plot(ax=ax, x_values=values["dim1"], y_values=values["dim2"], labels=labels, x_anchors=None)

        # set axislabels
        label_axes(ax, xlabel=f"PC{pc_x} ({var_dim1}%)", ylabel=f"PC{pc_y} ({var_dim2}%)")

    @classmethod
    def scree_plot(
        cls,
        adata: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        n_pcs: int = 20,
        dim_space: str = "obs",
        embbedings_name: str | None = None,
        scatter_kwargs: dict | None = None,
    ) -> None:
        """Plot the eigenvalues of each of the PCs using the scatter method

        Parameters
        ----------
        data : ad.AnnData
            AnnData to plot.
        ax : plt.Axes
            Matplotlib axes object to plot on.
        n_pcs : int,
            number of PCs to plot, by default 20
        dim_space : str, optional
            The dimension space used in PCA. Can be either "obs" (default) for sample projection or "var" for feature projection. By default "obs".
        embbedings_name : str | None, optional
            The custom embeddings name used in PCA. If None, uses default naming convention. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        # Generate the correct variance key name
        variance_key = f"variance_pca_{dim_space}" if embbedings_name is None else embbedings_name

        # Input checks
        _validate_scree_plot_input(adata, variance_key, n_pcs)

        n_pcs_avail = len(adata.uns[variance_key]["variance_ratio"])
        n_pcs = min(n_pcs, n_pcs_avail)

        # create the dataframe for plotting, X = pcs, y = explained variance
        values = pd.DataFrame(
            {
                "PC": np.arange(n_pcs) + 1,
                "explained_variance": adata.uns[variance_key]["variance_ratio"][:n_pcs],
            }
        )

        cls.scatter(
            data=values,
            x_column="PC",
            y_column="explained_variance",
            ax=ax,
            scatter_kwargs=scatter_kwargs,
        )

        # set labels
        space_suffix = " (samples)" if dim_space == "obs" else " (features)"
        label_axes(ax, xlabel="PC number", ylabel=f"Explained variance (%){space_suffix}")

    @classmethod
    def plot_pca_loadings(
        cls,
        data: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        dim_space: str = "obs",
        embbedings_name: str | None = None,
        dim: int = 1,
        nfeatures: int = 20,
        scatter_kwargs: dict | None = None,
    ) -> None:
        """Plot the gene loadings of a PC using the scatter method

        Parameters
        ----------
        data : ad.AnnData
            AnnData to plot.
        ax : plt.Axes
            Matplotlib axes object to plot on.
        dim_space : str, optional
            The dimension space used in PCA. Can be either "obs" (default) for sample projection or "var" for feature projection. By default "obs".
        embbedings_name : str | None, optional
            The custom embeddings name used in PCA. If None, uses default naming convention. By default None.
        dim : int
            The PC number from which to get loadings, by default 1 (1-indexed, i.e. the first PC is 1, not 0).
        nfeatures : int
            The number of top absolute loadings features to plot, by default 20
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        # Generate the correct loadings key name
        loadings_key = f"PCs_{dim_space}" if embbedings_name is None else embbedings_name

        # Determine which attribute to use for loadings based on dim_space
        loadings_attr = "varm" if dim_space == "obs" else "obsm"

        _validate_pca_loadings_plot_inputs(
            data=data, loadings_name=loadings_key, dim=dim, dim2=None, nfeatures=nfeatures, dim_space=dim_space
        )

        # create the dataframe for plotting
        dim_z = dim - 1  # to account from 0 indexing
        loadings_matrix = getattr(data, loadings_attr)[loadings_key]
        loadings = pd.DataFrame({"dim_loadings": loadings_matrix[:, dim_z]})

        # Use appropriate index for features based on dim_space
        if dim_space == "obs":
            loadings["feature"] = data.var.index.astype("string")
        else:  # dim_space == "var"
            loadings["feature"] = data.obs.index.astype("string")

        loadings["abs_loadings"] = loadings["dim_loadings"].abs()
        # Sort the DataFrame by absolute loadings and select the top features
        top_loadings = loadings.sort_values(by="abs_loadings", ascending=False).copy().head(nfeatures)
        top_loadings = top_loadings.reset_index(drop=True)
        top_loadings["index_int"] = range(nfeatures, 0, -1)

        cls.scatter(
            data=top_loadings,
            x_column="dim_loadings",
            y_column="index_int",
            ax=ax,
            scatter_kwargs=scatter_kwargs,
        )

        # set axis labels
        space_suffix = " features" if dim_space == "obs" else " samples"
        label_axes(ax, xlabel=f"PC{dim} loadings", ylabel=f"Top{space_suffix}")
        ax.set_yticks(top_loadings["index_int"])
        ax.set_yticklabels(top_loadings["feature"], rotation=0, ha="right")

    @classmethod
    def plot_pca_loadings_2d(
        cls,
        data: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        dim_space: str = "obs",
        embbedings_name: str | None = None,
        pc_x: int = 1,
        pc_y: int = 2,
        nfeatures: int = 20,
        *,
        add_labels: bool = True,
        add_lines: bool = False,
        scatter_kwargs: dict | None = None,
    ) -> None:
        """Plot the gene loadings of a PC using the scatter method

        Parameters
        ----------
        data : ad.AnnData
            AnnData to plot.
        ax : plt.Axes
            Matplotlib axes object to plot on.
        dim_space : str, optional
            The dimension space used in PCA. Can be either "obs" (default) for sample projection or "var" for feature projection. By default "obs".
        embbedings_name : str | None, optional
            The custom embeddings name used in PCA. If None, uses default naming convention. By default None.
        pc_x : int
            The PC principal component index to plot on the x axis, by default 1. Corresponds to the principal component order, the first principal is 1 (1-indexed, i.e. the first PC is 1, not 0).
        pc_y : int
            The principal component index to plot on the y axis, by default 2. Corresponds to the principal component order, the first principal is 1 (1-indexed, i.e. the first PC is 1, not 0).
        nfeatures : int
            The number of top absolute loadings features to label from each component, by default 20
        add_labels : bool
            Whether to add feature labels of the top `nfeatures` loadings. by default `True`.
        add_lines : bool
            If True, draw lines connecting the origin (0,0) to the points representing the top `nfeatures` loadings. Default is `False`.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        # Generate the correct loadings key name
        loadings_key = f"PCs_{dim_space}" if embbedings_name is None else embbedings_name

        loadings = _prepare_loading_df_to_plot(
            data=data, loadings_name=loadings_key, pc_x=pc_x, pc_y=pc_y, nfeatures=nfeatures, dim_space=dim_space
        )

        # plot the loadings of all features (used in PCA) first
        scatter_kwargs.update({"alpha": 0.3, "s": 10, "edgecolors": "none"})

        cls.scatter(
            data=loadings,
            x_column="dim1_loadings",
            y_column="dim2_loadings",
            ax=ax,
            color="grey",
            scatter_kwargs=scatter_kwargs,
        )

        loadings_top = loadings[loadings["is_top"]]

        # plot the top features on top
        scatter_kwargs.update({"alpha": 1, "s": 20, "edgecolors": "none"})

        cls.scatter(
            data=loadings_top,
            x_column="dim1_loadings",
            y_column="dim2_loadings",
            ax=ax,
            color="blue",
            scatter_kwargs=scatter_kwargs,
        )

        # add labels to the top features
        if add_labels:
            label_plot(
                ax=ax,
                x_values=loadings_top["dim1_loadings"],
                y_values=loadings_top["dim2_loadings"],
                labels=loadings_top["feature"],
                x_anchors=None,
                label_kwargs={"fontsize": config["font_sizes"]["medium"], "ha": "center", "va": "bottom"},
                line_kwargs={"color": BaseColors.get("black"), "linewidth": config["linewidths"]["medium"]},
            )
        # draw lines from the origin to the top features if specified
        if add_lines:
            for xi, yi in zip(loadings_top["dim1_loadings"], loadings_top["dim2_loadings"], strict=False):
                ax.plot([0, xi], [0, yi], color="gray", linestyle="-", linewidth=0.2)

        # set axis labels
        space_suffix = " (samples)" if dim_space == "obs" else " (features)"
        label_axes(ax, xlabel=f"PC{pc_x}{space_suffix}", ylabel=f"PC{pc_y}{space_suffix}")
