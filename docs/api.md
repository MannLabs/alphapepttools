# API

## Preprocessing

```{eval-rst}
.. module:: alphapepttools.pp
.. currentmodule:: alphapepttools

.. autosummary::
    :toctree: generated

    pp.add_metadata
    pp.filter_by_metadata
    pp.filter_data_completeness
    pp.scale_and_center
    pp.nanlog
    pp.detect_special_values
    pp.normalize
    pp.impute_gaussian
    pp.impute_median
    pp.impute_knn
    pp.impute_bpca
    pp.scanpy_pycombat
    pp.drop_singleton_batches

```

## Tools

```{eval-rst}
.. module:: alphapepttools.tl
.. currentmodule:: alphapepttools

.. autosummary::
    :toctree: generated

    tl.get_id2gene_map
    tl.map_genes_to_protein_groups
    tl.nan_safe_bh_correction
    tl.nan_safe_ttest_ind
    tl.diff_exp_ttest
    tl.diff_exp_alphaquant
    tl.diff_exp_ebayes
    tl.pca
    tl.bpca
    tl.extract_pca_anndata
    tl.prepare_pca_1d_loadings_data_to_plot
    tl.prepare_pca_2d_loadings_data_to_plot
    tl.prepare_scree_data_to_plot
    tl.tl_defaults
```

## Metrics

```{eval-rst}
.. module:: alphapepttools.metrics
.. currentmodule:: alphapepttools

.. autosummary::
    :toctree: generated

    metrics.coefficient_of_variation
    metrics.principal_component_regression
    metrics.pooled_coefficient_of_variation
    metrics.pooled_median_absolute_deviation
    metrics.calculate_qc_metrics
    metrics.fraction_complete
    metrics.number_detected
    metrics.total_intensity

```

## Plotting

```{eval-rst}
.. module:: alphapepttools.pl
.. currentmodule:: alphapepttools

.. autosummary::
    :toctree: generated

    pl.Plots
    pl.add_lines
    pl.label_plot
    pl.BaseColormaps
    pl.BaseColors
    pl.BasePalettes
    pl.MappedColormaps
    pl.show_rgba_color_list
    pl.PlotConfig
    pl.make_scatter_config
    pl.add_legend_to_axes
    pl.add_legend_to_axes_from_patches
    pl.create_figure
    pl.label_axes
    pl.save_figure
    pl.get_color_mapping
    pl.layered_plot
    pl.histogram
    pl.scatter
    pl.barplot
    pl.boxplot
    pl.violinplot
    pl.rank_median_plot
    pl.plot_pca
    pl.scree_plot
    pl.plot_pca_loadings
    pl.plot_pca_loadings_2d
    pl.volcano
```

## IO

### Reader functions

```{eval-rst}
.. module:: alphapepttools.io
.. currentmodule:: alphapepttools

.. autosummary::
    :toctree: generated

    io.read_psm_table
    io.read_pg_table
    io.AnnDataFactory
    io.list_available_reader
```

## Data

Example data that can be accessed with the package.

```{eval-rst}
.. module:: alphapepttools.data
.. currentmodule:: alphapepttools

.. autosummary::
    :toctree: generated

    data.available_data
    data.get_data
```
