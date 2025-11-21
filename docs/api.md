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
    pp.scanpy_pycombat

```

## Tools

```{eval-rst}
.. module:: alphapepttools.tl
.. currentmodule:: alphapepttools

.. autosummary::
    :toctree: generated

    tl.nan_safe_bh_correction
    tl.nan_safe_ttest_ind
    tl.diff_exp_ttest
    tl.diff_exp_alphaquant
    tl.pca
    tl.diff_exp_ebayes

```

## Metrics

```{eval-rst}
.. module:: alphapepttools.metrics
.. currentmodule:: alphapepttools

.. autosummary::
    :toctree: generated

    metrics.principal_component_regression
    metrics.pooled_median_absolute_deviation
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
    pl.add_legend_to_axes
    pl.add_legend_to_axes_from_patches
    pl.create_figure
    pl.label_axes
    pl.save_figure

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
    io.available_reader
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
