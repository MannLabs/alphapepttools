# Changelog

All notable changes to this project will be documented on the
[https://github.com/MannLabs/alphapepttools/releases](GitHub Releases pages).

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- `tl.pca` and `tl.bpca` record the samples and features the PCA was fitted on as `obs_names` and `var_names` in their `adata.uns` entry
