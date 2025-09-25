# `alphatools`

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/MannLabs/alphatools/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/alphatools

Search- and quantification-engine agnostic downstream processing of proteomics data

## `alphatools` was made to make your life in proteomics easier!

Functionalities are intended to be as close to pure python as possible, avoiding closed end-to-end implementations, which is reflected in several design choices:

1. AnnData is used in favor of a custom data class to enable interoperability with any other tool from the Scverse.
2. matplotlib _Axes_ and _Figure_ instances are used for visualization, giving the user full autonomy to layer on custom visualizations with searborn, matplotlib, or any other compatible visualization package.
3. Statistical and preprocessing functions are standalone and set with strong defaults, meaning that any function can be used outside of the `alphatools` context.

### Design choices of `alphatools`:

- **Data handling**: `AnnData` was chosen as a data container for two main reasons:
    1. For presenting a lightweight, powerful solution to a fundamental challenge with dataframes, which is keeping numerical data and metadata aligned together at all times. Using dataframes, the options are to either include non-numeric metadata columns in the dataframe (complicating data operations) or to add cumbersome multi-level indices and
    2. For their compatibility with the Scverse, Scanpy and all associated tools, essentially removing the barrier between proteomics and transcriptomics data analysis and enabling multi-omics analyses.
       <br>

- **Plotting**: Inspired by the [`stylia`] package, `alphatools` aims to provide a consistent and aesthetically pleasing visual experience for all plots. A core component of this implementation is the fact that `create_figure` returns subplots as an iterable data structure, meaning that once the basic layout of a plot is decided, users simply jump from one plot window to the next and populate each one with figure elements.
- **Standardization**: A key consideration of this package is the loading of proteomics data, the biggest painpoint of which is the nonstandard output of various proteomic search engines. By building on `alphabase`, we handle this complexity early and provide the user with AnnData objects containing either proteins or precursors, where the familiar Pandas DataFrame is always just a '`df = adata.to_df().join(adata.obs)`' away.

[`stylia`]: https://github.com/ersilia-os/stylia.git

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [Mambaforge][].

There are several alternative options to install alphatools:

<!--
1) Install the latest release of `alphatools` from [PyPI][]:

```bash
pip install alphatools
```
-->

1. Install the latest development version:

```bash
git clone git+https://github.com/MannLabs/alphatools.git@main && cd alphatools
pip install -e .
```

or with more dependencies:

```bash
pip install -e ".[test, dev]"
```

## Release notes

See the [GitHub Release page](https://github.com/MannLabs/alphatools/releases).

## Developer Guide

This document gathers information on how to develop and contribute to the alphaDIA project.

### Release process

#### Tagging of changes

In order to have release notes automatically generated, changes need to be tagged with labels.
The following labels are used (should be safe-explanatory):
`breaking-change`, `bug`, `enhancement`.

#### Release a new version

This package uses a shared release process defined in the
[alphashared](https://github.com/MannLabs/alphashared) repository. Please see the instructions
[there](https://github.com/MannLabs/alphashared/blob/reusable-release-workflow/.github/workflows/README.md#release-a-new-version)

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/MannLabs/alphatools/issues
[tests]: https://github.com/MannLabs/alphatools/actions/workflows/test.yml
[documentation]: https://alphatools.readthedocs.io
[changelog]: https://alphatools.readthedocs.io/en/latest/changelog.html
[api documentation]: https://alphatools.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/alphatools
