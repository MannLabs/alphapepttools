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

### Changed

- `tl.pca` and `tl.bpca` always compute the projection of the observations (samples). For a projection of the features, pass the transposed object: `tl.pca(adata=adata.T)`
- The `meta_data_mask_column_name` parameter of `tl.pca` and `tl.bpca` is renamed to `feature_mask_column`
- The default result keys lost their `_obs` suffix: `X_pca`, `PCs_pca`, `variance_pca` and `X_bpca`, `PCs_bpca`, `variance_bpca`. The defaults of `metrics.principal_component_regression` follow. Objects saved with the old keys need PCA to be re-run before plotting

### Removed

- The `dim_space` parameter of `tl.pca`, `tl.bpca`, `tl.extract_pca_anndata`, `tl.prepare_scree_data_to_plot`, `tl.prepare_pca_1d_loadings_data_to_plot`, `tl.prepare_pca_2d_loadings_data_to_plot`, `pl.plot_pca`, `pl.scree_plot`, `pl.plot_pca_loadings` and `pl.plot_pca_loadings_2d`
