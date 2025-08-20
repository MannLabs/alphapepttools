# alphatools

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/MannLabs/alphatools/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/alphatools

Search- and quantification-engine agnostic biological interpretation of proteomics data

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
