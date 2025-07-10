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

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from alphatools.pl import defaults
from alphatools.pl.colors import BaseColors, BasePalettes, get_color_mapping
from alphatools.pl.figure import create_figure, label_axes
from alphatools.pp.data import _adata_column_to_array

# logging configuration
logging.basicConfig(level=logging.INFO)

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


def _make_legend_patches(
    color_dict: dict[str, str | tuple],
) -> list[mpl.patches.Patch]:
    """Create legend patches for a matplotlib legend from a value-to-color mapping"""
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


def make_legend(
    ax: plt.Axes,
    patches: list[mpl.patches.Patch],
    **kwargs,
) -> None:
    """Add a legend to a matplotlib axes object

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

    # handle the title separately
    _legend.set_title(_legend.get_title().get_text(), prop={"size": config["legend"]["title_size"]})


def add_legend(
    ax: plt.Axes,
    levels: list[str] | None,
    palette: list[str | tuple] | None,
    legend: str | mpl.legend.Legend | None = None,
    **legend_kwargs,
) -> None:
    """Add a legend to an axis object.

    If levels and palette are provided and legend is None, a legend is created automatically.
    if legend is set to "auto", a legend is created from the levels and a default palette.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to add the legend to.
    levels : list[str] | None
        List of levels to use for the legend. Duplicates are removed.
    palette : list[str | tuple] | None
        List of colors to use for the legend. If None, a default palette will be used. By default None.

    """
    if legend not in ["auto", None]:
        raise ValueError("legend must be 'auto' or None")

    if levels is None:
        logging.warning("No levels provided. Skipping legend creation.")
        return

    levels = np.unique(levels)

    # Determine palette, i.e. list of colors to show in the legend
    if palette is None:
        if legend == "auto":
            palette = BasePalettes.get("qualitative")
        else:
            raise ValueError("Palette must be provided if legend is not set to 'auto'")

    color_dict = get_color_mapping(levels, palette)
    patches = _make_legend_patches(color_dict)
    make_legend(ax, patches, **legend_kwargs)


def _drop_nans_from_plot_arrays(
    x_values: np.ndarray,
    y_values: np.ndarray,
    labels: np.ndarray | list[str],
) -> tuple:
    # Missing x or y values are breaking and should be dropped
    keep_mask = ~np.logical_or(np.isnan(x_values), np.isnan(y_values))

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
    data: ad.AnnData, pca_embeddings_layer_name: str, pca_variance_layer_name: str, pc_x: int, pc_y: int
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
    """
    if not isinstance(data, ad.AnnData):
        raise TypeError("data must be an AnnData object")

    if pca_embeddings_layer_name not in data.obsm:
        raise ValueError(
            f"PCA embeddings layer '{pca_embeddings_layer_name}' not found in AnnData object. "
            f"Found layers: {list(data.obsm.keys())}"
        )

    if pca_variance_layer_name not in data.uns:
        raise ValueError(
            f"PCA metadata layer '{pca_variance_layer_name}' not found in AnnData object. "
            f"Found layers: {list(data.uns.keys())}"
        )

    n_pcs = data.obsm[pca_embeddings_layer_name].shape[1]
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
    data: ad.AnnData, loadings_name: str, dim: int, dim2: int | None, nfeatures: int
) -> None:
    """
    Validate inputs for accessing PCA feature loadings from an AnnData object.

    Parameters
    ----------
    data : anndata.AnnData
        The AnnData object containing PCA loadings data.
    loadings_name : str
        The key in `data.varm` that stores PCA feature loadings (e.g., "PCs").
    dim : int
        The principal component index (1-based) to extract loadings for.
    dim2 : int | None
        The second principal component index (1-based) to extract loadings for, if applicable.
    nfeatures : int
        The number of top features to consider for the given component.
    """
    if not isinstance(data, ad.AnnData):
        raise TypeError("data must be an AnnData object")

    if loadings_name not in data.varm:
        raise ValueError(
            f"PCA feature loadings layer '{loadings_name}' not found in AnnData object. "
            f"Found layers: {list(data.varm.keys())}"
        )

    n_pcs = data.varm[loadings_name].shape[1]
    if not (1 <= dim <= n_pcs):
        raise ValueError(f"PC must be between 1 and {n_pcs} (inclusive). Got dim={dim}.")
    if dim2 is not None and not (1 <= dim2 <= n_pcs):
        raise ValueError(f"second PC must be between 1 and {n_pcs} (inclusive). Got pc_y={dim2}.")

    n_features = data.varm[loadings_name].shape[0]
    if not (1 <= nfeatures <= n_features):
        raise ValueError(
            f"Number of features must be between 1 and {n_features} (inclusive). Got nfeatures={nfeatures}."
        )


def _prepare_loading_df_to_plot(
    data: ad.AnnData,
    loadings_name: str,
    pc_x: int,
    pc_y: int,
    nfeatures: int,
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
        The key in `data.varm` where PCA loadings are stored.
    pc_x : int
        The first principal component index (1-based) to extract loadings for.
    pc_y : int
        The second principal component index (1-based) to extract loadings for.
    nfeatures : int
        Number of top features per PC to highlight based on absolute loadings.

    Returns
    -------
    pd.DataFrame
        DataFrame containing loadings for the selected PCs, feature names, boolean columns
        indicating if a feature was used in PCA and whether it is among the top features in either dimension.
    """
    dim1_z = pc_x - 1  # convert to 0-based index
    dim2_z = pc_y - 1  # convert to 0-based index

    orig_loadings = data.varm[loadings_name]

    loadings = pd.DataFrame(
        {
            "dim1_loadings": orig_loadings[:, dim1_z],
            "dim2_loadings": orig_loadings[:, dim2_z],
        }
    )

    # Add feature names and absolute loadings
    loadings["feature"] = data.var_names

    # get only features that were used in the PCA (e.g., those that are part of the core proteome)
    # these would be features with 0 loadings in all PC dimensions
    loading_sums = np.nansum(orig_loadings, axis=1)
    non_sum_zero = np.where(loading_sums != 0)[0]
    is_in_pca = np.zeros(data.n_vars, dtype=bool)
    is_in_pca[non_sum_zero] = True
    loadings["is_in_pca"] = is_in_pca

    # filter the loadings to only include features that were used in the PCA
    loadings = loadings[loadings["is_in_pca"]]

    # add the top N features for each dimension
    loadings["abs_dim1"] = loadings["dim1_loadings"].abs()
    loadings["abs_dim2"] = loadings["dim2_loadings"].abs()

    loadings["is_top"] = False
    loadings.loc[loadings.nlargest(nfeatures, "abs_dim1").index, "is_top"] = True
    loadings.loc[loadings.nlargest(nfeatures, "abs_dim2").index, "is_top"] = True

    return loadings


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
        color_column: str | None = None,
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
        color_column : str, optional
            Column in data to use for color encoding. Overrides color parameter. By default None.
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

        if not ax:
            _, ax = create_figure(1, 1)

        values = _adata_column_to_array(data, value_column)

        if color_column is None:
            color = BaseColors.get(color)
            ax.hist(values, bins=bins, color=color, **hist_kwargs)
        else:
            color_values = _adata_column_to_array(data, color_column)
            palette = palette or BasePalettes.get("qualitative")
            color_dict = color_dict or get_color_mapping(color_values, palette)
            missing = set(np.unique(color_values)) - set(color_dict)
            for level in missing:
                color_dict[level] = BaseColors.get("grey")

            for level, level_color in color_dict.items():
                ax.hist(
                    values[color_values == level],
                    bins=bins,
                    color=level_color,
                    **hist_kwargs,
                )

            if legend is not None:
                add_legend(
                    ax=ax,
                    levels=list(color_dict.keys()),
                    palette=list(color_dict.values()),
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
        y_column: str,
        x_column: str,
        color: str = "blue",
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

        Parameters
        ----------
        data : pd.DataFrame | ad.AnnData
            Data to plot, must contain the x_column and y_column and optionally the color_column.
        x_column : str
            Column in data to plot on the x-axis. Must contain numeric data.
        y_column : str
            Column in data to plot on the y-axis. Must contain numeric data.
        color_column : str, optional
            Column in data to use for color encoding. By default None.
        ax : plt.Axes, optional
            Matplotlib axes object to plot on, if None a new figure is created. By default None.
        palette : list[str | tuple], optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
        color_dict: dict[str, str | tuple], optional
            Supercedes palette, a dictionary mapping levels to colors. By default None. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.
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

        if not ax:
            _, axm = create_figure()
            ax = axm.next()

        if color_column is None:
            color_values = ["data"] * len(data)
            color_dict = {"data": BaseColors.get(color)}
        else:
            color_values = _adata_column_to_array(data, color_column)
            palette = palette or BasePalettes.get("qualitative")
            color_dict = color_dict or get_color_mapping(color_values, palette)
            missing = set(np.unique(color_values)) - set(color_dict)
            for level in missing:
                color_dict[level] = BaseColors.get("grey")

        # Handle ordering of plotting arrays: order by the frequency of the color column
        counts = Counter(color_values)
        order = np.argsort([counts[cv] for cv in color_values])[::-1]
        x_values = _adata_column_to_array(data, x_column)[order]
        y_values = _adata_column_to_array(data, y_column)[order]
        color_values = np.array(color_values)[order]

        ax.scatter(
            x=x_values,
            y=y_values,
            c=[color_dict[color] for color in color_values],
            **scatter_kwargs,
        )

        if legend is not None:
            add_legend(
                ax=ax,
                levels=list(color_dict.keys()),
                palette=list(color_dict.values()),
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
        palette: list[str | tuple] | None = None,
        color_dict: dict[str, str | tuple] | None = None,
        legend: str | mpl.legend.Legend | None = None,
        color_column: str | None = None,
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
        color_column : str, optional
            Column in data.var to use for color coding. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

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
        data: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        pc_x: int = 1,
        pc_y: int = 2,
        pca_embeddings_layer_name: str = "X_pca",
        pca_variance_layer_name: str = "pca",
        label: bool = False,  # noqa: FBT001, FBT002
        label_column: str | None = None,
        color: str = "blue",
        palette: list[str | tuple] | None = None,
        color_dict: dict[str, str | tuple] | None = None,
        legend: str | mpl.legend.Legend | None = None,
        color_column: str | None = None,
        scatter_kwargs: dict | None = None,
    ) -> None:
        """Plot the PCs of a PCA analysis using the scatter method

        Parameters
        ----------
        data : ad.AnnData
            AnnData to plot.
        ax : plt.Axes
            Matplotlib axes object to plot on.
        pc_x : int
            The PC principal component index to plot on the x axis, by default 1. Corresponds to the principal component order, the first principal is 1 (1-indexed, i.e. the first PC is 1, not 0).
        pc_y : int
            The principal component index to plot on the y axis, by default 2. Corresponds to the principal component order, the first principal is 1 (1-indexed, i.e. the first PC is 1, not 0).
        pca_embeddings_layer_name : str,
            The name of the PCA layer in the AnnData object (in `data.obsm`), by default "X_pca". Different name should be used in case `key_added` was specifically set in `pca()` function under `**pca_kwargs`.
        label: bool,
            The name of the PCA layer in the AnnData object (in `data.uns`) that contains the explained variance, by default "pca". Different name should be used in case `key_added` was specifically set in `pca()` function under `**pca_kwargs`.
        label: bool,
            Whether to add labels to the points in the scatter plot. by default False.
        label_column: str | None = None,
            Column in data.obs to use for labeling the points. If None, and label is True, data.obs.index labels are added. By default None.
        color : str, optional
            Color to use for the scatter plot. By default "blue".
        palette : list[str | tuple], optional
            List of colors to use for color encoding, if None a default palette is used. By default None.
        color_dict: dict[str, str | tuple], optional
            Supercedes palette, a dictionary mapping levels to colors. By default None. If provided, palette is ignored.
        legend : str | mpl.legend.Legend, optional
            Legend to add to the plot, by default None. If "auto", a legend is created from the color_column. By default None.
        color_column : str, optional
            Column in data.var to use for color coding. By default None.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        # Input checks
        _validate_pca_plot_input(data, pca_embeddings_layer_name, pca_variance_layer_name, pc_x, pc_y)

        # create the dataframe for plotting
        dim1_z = pc_x - 1  # to account for 0 indexing
        dim2_z = pc_y - 1  # to account for 0 indexing
        values = pd.DataFrame(data.obsm[pca_embeddings_layer_name][:, [dim1_z, dim2_z]], columns=["dim1", "dim2"])

        # get the explained variance ratio for the dimensions
        var_dim1 = data.uns[pca_variance_layer_name]["variance_ratio"][dim1_z]
        var_dim1 = round(var_dim1 * 100, 2)
        var_dim2 = data.uns[pca_variance_layer_name]["variance_ratio"][dim2_z]
        var_dim2 = round(var_dim2 * 100, 2)

        # add color column
        if color_column is not None:
            color_values = _adata_column_to_array(data, color_column)
            values[color_column] = color_values

        cls.scatter(
            data=values,
            x_column="dim1",
            y_column="dim2",
            color=color,
            color_column=color_column,
            legend=legend,
            palette=palette,
            color_dict=color_dict,
            ax=ax,
            scatter_kwargs=scatter_kwargs,
        )

        # add labels if requested
        if label:
            labels = data.obs.index if label_column is None else _adata_column_to_array(data, label_column)
            label_plot(ax=ax, x_values=values["dim1"], y_values=values["dim2"], labels=labels, x_anchors=None)

        # set axislabels
        label_axes(ax, xlabel=f"PC{pc_x} ({var_dim1}%)", ylabel=f"PC{pc_y} ({var_dim2}%)")

    @classmethod
    def scree_plot(
        cls,
        data: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        n_pcs: int = 20,
        pca_variance_layer_name: str = "pca",
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
        pca_variance_layer_name : str,
            The name of the PCA layer in the AnnData object (in `data.uns`) that contains the explained variance, by default "pca". Different name should be used in case `key_added` was specifically set in `pca()` function under `**pca_kwargs`.
        scatter_kwargs : dict, optional
            Additional keyword arguments for the matplotlib scatter function. By default None.

        Returns
        -------
        None

        """
        scatter_kwargs = scatter_kwargs or {}

        # Input checks
        _validate_scree_plot_input(data, pca_variance_layer_name, n_pcs)

        n_pcs_avail = len(data.uns[pca_variance_layer_name]["variance_ratio"])
        n_pcs = min(n_pcs, n_pcs_avail)

        # create the dataframe for plotting, X = pcs, y = explained variance
        values = pd.DataFrame(
            {
                "PC": np.arange(n_pcs) + 1,
                "explained_variance": data.uns[pca_variance_layer_name]["variance_ratio"][:n_pcs],
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
        label_axes(ax, xlabel="PC number", ylabel="Explained variance (%)")

    @classmethod
    def plot_pca_loadings(
        cls,
        data: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        loadings_name: str = "PCs",
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
        loadings_name : str
            The name of the PCA loadings layer in the AnnData object (data.varm.keys), by default "PCs". Different name should be used in case `key_added` was specifically set in `pca()` function under `**pca_kwargs`.
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

        _validate_pca_loadings_plot_inputs(
            data=data, loadings_name=loadings_name, dim=dim, dim2=None, nfeatures=nfeatures
        )

        # create the dataframe for plotting
        dim_z = dim - 1  # to account from 0 indexing
        loadings = pd.DataFrame({"dim_loadings": data.varm[loadings_name][:, dim_z]})
        loadings["feature"] = data.var.index.astype("string")
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
        label_axes(ax, xlabel=f"PC{dim} loadings", ylabel="Top features")
        ax.set_yticks(top_loadings["index_int"])
        ax.set_yticklabels(top_loadings["feature"], rotation=0, ha="right")

    @classmethod
    def plot_pca_loadings_2d(
        cls,
        data: ad.AnnData | pd.DataFrame,
        ax: plt.Axes,
        loadings_name: str = "PCs",
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
        loadings_name : str
            The name of the PCA loadings layer in the AnnData object (data.varm.keys), by default "PCs". Different name should be used in case `key_added` was specifically set in `pca()` function under `**pca_kwargs`.
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

        _validate_pca_loadings_plot_inputs(
            data=data, loadings_name=loadings_name, dim=pc_x, dim2=pc_y, nfeatures=nfeatures
        )

        loadings = _prepare_loading_df_to_plot(
            data=data, loadings_name=loadings_name, pc_x=pc_x, pc_y=pc_y, nfeatures=nfeatures
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
        label_axes(ax, xlabel=f"PC{pc_x}", ylabel=f"PC{pc_y}")
