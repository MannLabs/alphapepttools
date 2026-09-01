# Contributing guide

Scanpy provides extensive [developer documentation][scanpy developer guide], most of which applies to this project, too.
This document will not reproduce the entire content from there.
Instead, it aims at summarizing the most important information to get you started on contributing.

We assume that you are already familiar with git and with making pull requests on GitHub.
If not, please refer to the [scanpy developer guide][].

[scanpy developer guide]: https://scanpy.readthedocs.io/en/latest/dev/index.html

## Installing dev dependencies

In addition to the packages needed to _use_ this package,
you need additional python packages to [run tests](#writing-tests) and [build the documentation](#docs-building).

:::::{tabs}
::::{group-tab} Hatch
The easiest way is to get familiar with [hatch environments][], with which these tasks are simply:

```bash
hatch test  # defined in the table [tool.hatch.envs.hatch-test] in pyproject.toml
hatch run docs:build  # defined in the table [tool.hatch.envs.docs]
```

::::

::::{group-tab} Pip
If you prefer managing environments manually, you can use `pip`:

```bash
cd alphapepttools
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test,doc]"
```

::::
:::::

[hatch environments]: https://hatch.pypa.io/latest/tutorials/environment/basic-usage/

## Handling anndata objects

The central data structure of `alphapepttools` is the `anndata.AnnData` object. All functions should be compatible with `anndata.AnnData`. We default to `inplace` modifications, i.e., functions should default to returning `None` and acting directly on the passed `anndata.AnnData` object. This behaviour is adapted from [`scanpy v1`](https://scanpy.readthedocs.io/en/stable/) and aims to maximize the compatibility of the interfaces.

### Acting on omics measurements

Functions that act on the measurement data (`.X`, `.layers[...]`) in the anndata object (typically in the `.pp` and `.tl` modules) or change the object's shape (e.g., filtering operations) _MUST_ use the following call signature:

```python
alphapepttools.pp.func(adata: ad.AnnData, ..., *, layer: str | None = None, copy: bool = False) -> None | ad.AnnData:
...

alphapepttools.tl.func(adata: ad.AnnData, ..., *, layer: str | None = None, copy: bool = False) -> None | ad.AnnData:
  ...

```

**Layer modification** This means that they take an `anndata.AnnData` object and modify/update a specific measurement layer in the object. Per default (`None`), this will be the `anndata.AnnData.X` attribute, otherwise the specified layer.

**Modification inplace** Per default, the `anndata.AnnData` object is modified inplace (`copy=False`), this means that the current object is updated and the function returns `None`. If `copy=True`, an updated copy of the object is returned and the original object remains unchanged.

#### Examples

Default behaviour:

```python
adata.layers["original"] = adata.X.copy()

return_value = alphapepttools.pp.func(adata)
assert return_value is None
assert not np.array_equal(adata.X, adata.layers["original"])
```

Act on a specific layer

```python
adata.layers["original"] = adata.X.copy()
adata.layers["new_layer"] = adata.X.copy()

return_value = alphapepttools.pp.func(adata, layer="new_layer")
assert return_value is None

# adata.X is unchanged
assert np.array_equal(adata.X, adata.layers["original"])

# New layer is changed
assert not np.array_equal(adata.layers["new_layer"], adata.layers["original"])
```

Return an updated copy

```python
adata_original = adata.copy()
adata_new = alphapepttools.pp.func(adata, copy=True)
# Returns an updated anndata object
assert not np.array_equal(adata.X, adata_new.X)

# The original anndata remains unchanged
assert np.array_equal(adata.X, adata_original.X)
```

### Generating summary statistics from anndata object

Functions that generate new summary statistics and do not act on the measurement layers `.X`/`.layers` (e.g., differential expression analysis results or summary metrics; typically in the `.tl` and `.metrics` modules) _SHOULD_ store the results inplace in the anndata object. They _MAY_ also allow users to return the results as a `pandas.DataFrame` by exposing an `inplace` argument. They should follow this general call signature:

```python
alphapepttools.tl.func(adata: ad.AnnData, ..., *, layer: str | None = None, inplace: bool = True) -> None | pd.DataFrame:
  ...

alphapepttools.metrics.func(adata: ad.AnnData, ..., *, layer: str | None = None, inplace: bool = True) -> None | pd.DataFrame:
  ...
```

Similarly to the `copy` argument, the keyword `inplace` determines whether the anndata object is directly modified or not. `inplace` should always be used when the result is either added to a non-measurement slot of the anndata object (`.obs`, `.var`, `.obsm`, `.varm`, `.uns`) or returned as a `pandas.DataFrame`. In contrast, `copy` should be used when a measurement slot is modified.

#### Examples

```python
result = apt.tl.func(adata, ..., inplace=False)
assert isinstance(result, pd.DataFrame)

result = apt.tl.func(adata, ..., inplace=True)
assert result is None
```

## Code-style

This package uses [pre-commit][] to enforce consistent code-styles.
On every commit, pre-commit checks will either automatically fix issues with the code, or raise an error message.

To enable pre-commit locally, simply run

```bash
pre-commit install
```

in the root of the repository.
Pre-commit will automatically download all dependencies when it is run for the first time.

<!--
DECISION: we don't want to use pre-commit.ci for the moment.

Alternatively, you can rely on the [pre-commit.ci][] service enabled on GitHub.
If you didn't run `pre-commit` before pushing changes to GitHub it will automatically commit fixes to your pull request, or show an error message.

If pre-commit.ci added a commit on a branch you still have been working on locally, simply use

```bash
git pull --rebase
```

to integrate the changes into yours.
While the [pre-commit.ci][] is useful, we strongly encourage installing and running pre-commit locally first to understand its usage.

[pre-commit.ci]: https://pre-commit.ci/
-->

Finally, most editors have an _autoformat on save_ feature.
Consider enabling this option for [ruff][ruff-editors] and [prettier][prettier-editors].

[pre-commit]: https://pre-commit.com/
[ruff-editors]: https://docs.astral.sh/ruff/integrations/
[prettier-editors]: https://prettier.io/docs/en/editors.html

## Writing modules

Top-level modules consist of `data`, `io`, `metrics`, `pl`, `pp` and `tl`. Modules are usually python scripts containing semantically related code (e.g. code to impute values lives in the `impute.py` module inside `pp`). If necessary, sub-module directories may be introduced to enhance clarity, but in general a flat structure is preferred. `.py` modules are lowercase and don't start with underscores, and generally have a `test_....py` module in the corresponding tests-top-level module. An example for this structure is the top-level module `io/anndata_factory.py`, which is covered by unit tests in `tests/io/test_anndata_factory.py`.

## Writing tests

This package uses [pytest][] for automated testing.
Please write {doc}`scanpy:dev/testing` for every function added to the package.

Most IDEs integrate with pytest and provide a GUI to run tests.
Just point yours to one of the environments returned by

```bash
hatch env create hatch-test  # create test environments for all supported versions
hatch env find hatch-test  # list all possible test environment paths
```

Alternatively, you can run all tests from the command line by executing

:::::{tabs}
::::{group-tab} Hatch

```bash
hatch test  # test with the highest supported Python version
# or
hatch test --all  # test with all supported Python versions
```

::::

::::{group-tab} Pip

```bash
source .venv/bin/activate
pytest
```

::::
:::::

in the root of the repository.

[pytest]: https://docs.pytest.org/

### Continuous integration

Continuous integration will automatically run the tests on all pull requests and test
against the minimum and maximum supported Python version.

Additionally, there's a CI job that tests against pre-releases of all dependencies (if there are any).
The purpose of this check is to detect incompatibilities of new package versions early on and
gives you time to fix the issue or reach out to the developers of the dependency before the package is released to a wider audience.

## Publishing a release

### Updating the version number

Before making a release, you need to update the version number in the `pyproject.toml` file.
Please adhere to [Semantic Versioning][semver], in brief

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
> 1. MAJOR version when you make incompatible API changes,
> 2. MINOR version when you add functionality in a backwards compatible manner, and
> 3. PATCH version when you make backwards compatible bug fixes.
>
> Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

Once you are done, commit and push your changes and navigate to the "Releases" page of this project on GitHub.
Specify `vX.X.X` as a tag name and create a release.
For more information, see [managing GitHub releases][].
This will automatically create a git tag and trigger a Github workflow that creates a release on [PyPI][].

[semver]: https://semver.org/
[managing GitHub releases]: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
[pypi]: https://pypi.org/

## Writing documentation

Please write documentation for new or changed features and use-cases.
This project uses [sphinx][] with the following features:

- The [myst][] extension allows to write documentation in markdown/Markedly Structured Text
- [Numpy-style docstrings][numpydoc] (through the [napoloen][numpydoc-napoleon] extension).
- The last sentence in a docstring should not have a trailing punctuation mark
- Type hints should not be added to parameters or returns (automatically added by the sphinx docstring compiler)
- Docstrings should contain examples written as Python code blocks in restructured-text format (https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- Jupyter notebooks as tutorials through [myst-nb][] (See [Tutorials with myst-nb](#tutorials-with-myst-nb-and-jupyter-notebooks))
- [sphinx-autodoc-typehints][], to automatically reference annotated input and output types
- Citations (like {cite:p}`Virshup_2023`) can be included with [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/)

See scanpy’s {doc}`scanpy:dev/documentation` for more information on how to write your own.

[sphinx]: https://www.sphinx-doc.org/en/master/
[myst]: https://myst-parser.readthedocs.io/en/latest/intro.html
[myst-nb]: https://myst-nb.readthedocs.io/en/latest/
[numpydoc-napoleon]: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
[sphinx-autodoc-typehints]: https://github.com/tox-dev/sphinx-autodoc-typehints

### Tutorials with myst-nb and jupyter notebooks

The documentation is set-up to render jupyter notebooks stored in the `docs/notebooks` directory using [myst-nb][].
Currently, only notebooks in `.ipynb` format are supported that will be included with both their input and output cells.
It is your responsibility to update and re-run the notebook whenever necessary.

If you are interested in automatically running notebooks as part of the continuous integration,
please check out [this feature request][issue-render-notebooks] in the `cookiecutter-scverse` repository.

[issue-render-notebooks]: https://github.com/scverse/cookiecutter-scverse/issues/40

#### Hints

- If you refer to objects from other packages, please add an entry to `intersphinx_mapping` in `docs/conf.py`.
  Only if you do so can sphinx automatically create a link to the external documentation.
- If building the documentation fails because of a missing link that is outside your control,
  you can add an entry to the `nitpick_ignore` list in `docs/conf.py`

(docs-building)=

#### Building the docs locally

:::::{tabs}
::::{group-tab} Hatch

```bash
hatch run docs:build
hatch run docs:open
```

::::

::::{group-tab} Pip

```bash
source .venv/bin/activate
cd docs
make html
(xdg-)open _build/html/index.html
```

::::
:::::
