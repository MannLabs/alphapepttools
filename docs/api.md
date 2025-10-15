# API

## Preprocessing

```{eval-rst}
.. module:: alphatools.pp
.. currentmodule:: alphatools

.. autosummary::
    :toctree: generated

    pp.add_metadata
    pp.filter_by_metadata
    pp.filter_data_completeness
    pp.load_diann_pg_matrix
    pp.scale_and_center
    pp.nanlog
    pp.detect_special_values
    pp.normalize
    pp.pca
    pp.impute
    pp.impute_gaussian
    pp.impute_median
    pp.add_core_proteome_mask
    pp.scanpy_pycombat

```

## Tools

```{eval-rst}
.. module:: alphatools.tl
.. currentmodule:: alphatools

.. autosummary::
    :toctree: generated

    tl.nan_safe_bh_correction
    tl.nan_safe_ttest_ind
    tl.diff_exp_ttest
    tl.diff_exp_alphaquant

```

## Metrics

```{eval-rst}
.. module:: alphatools.metrics
.. currentmodule:: alphatools

.. autosummary::
    :toctree: generated

    metrics.principal_component_regression
    metrics.pooled_median_absolute_deviation
```

## Plotting

```{eval-rst}
.. module:: alphatools.pl
.. currentmodule:: alphatools

.. autosummary::
    :toctree: generated

    pl.Plots
    pl.add_lines
    pl.label_plot
```

## IO

### Reader functions

```{eval-rst}
.. module:: alphatools.io
.. currentmodule:: alphatools

.. autosummary::
    :toctree: generated

    io.read_psm_table
    io.read_pg_table
    io.AnnDataFactory
```
