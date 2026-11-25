"""Deprecated helpers, scheduled for removal in version 0.7.0."""

import warnings

import anndata as ad


def drop_features_with_too_few_valid_values(
    adata: ad.AnnData,
    between_column: str,
    comparison: tuple[str, str],
    min_valid_values: int,
) -> ad.AnnData:
    """Drop features with too few valid (non-NaN) values in either group.

    .. deprecated:: 0.6.0
        Will be removed in version 0.7.0. Use :func:`alphapepttools.pp.filter_data_completeness`
        with `group_column`, `groups` and `action="drop"` instead.

    Parameters
    ----------
    adata
        The AnnData object containing the data.
    between_column
        The column name in adata.obs to use for grouping.
    comparison
        The two groups to compare.
    min_valid_values
        The minimum number of valid (non-NaN) values required.

    Returns
    -------
    ad.AnnData
        A copy of the filtered AnnData object.

    Examples
    --------
    .. code-block:: python

        import anndata as ad
        import numpy as np
        from alphapepttools.tl.utils import drop_features_with_too_few_valid_values

        # Create data with some missing values
        X = np.array(
            [
                [1, 2, np.nan, 4],
                [5, np.nan, np.nan, 8],
                [9, 10, 11, 12],
                [np.nan, 14, 15, 16],
                [17, 18, np.nan, 20],
                [21, 22, 23, 24],
            ]
        )

        adata = ad.AnnData(
            X=X, obs={"group": ["A", "A", "A", "B", "B", "B"]}, var={"feature": ["F1", "F2", "F3", "F4"]}
        )

        # Keep only features with at least 2 valid values per group
        filtered = drop_features_with_too_few_valid_values(adata, "group", ("A", "B"), min_valid_values=2)

        # in this case, features 1, 2 and 4 would be kept, while feature 3 would be dropped due to too many NaNs

    """
    warnings.warn(
        "`drop_features_with_too_few_valid_values` is deprecated and will be removed in version 0.7.0. "
        "Use `alphapepttools.pp.filter_data_completeness` instead.",
        FutureWarning,
        stacklevel=2,
    )
    for lvl in comparison:
        valid_counts = adata[adata.obs[between_column] == lvl].to_df().astype(float).count(axis=0)
        features_to_keep = valid_counts >= min_valid_values
        adata = adata[:, features_to_keep].copy()

    return adata
