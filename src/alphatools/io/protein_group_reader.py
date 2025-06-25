from typing import Literal

import anndata as ad
import pandas as pd


def _validate_row_type(row_type: Literal["observations", "features"]) -> None:
    """Raise KeyError if row type is not one of expected values"""
    if row_type not in ("features", "observations"):
        raise KeyError("`row_type` must be one of `features` or `observations`")


def read_pg_matrix(
    file_path: str,
    row_type: Literal["observations", "features"] = "features",
    separator: str = "\t",
    sample_metadata_index: str | int | list[str] | list[int] = 0,
    feature_metadata_index: str | int | list[str] | list[int] = 0,
    sample_name: str | list[str] | None = None,
    feature_name: str | list[str] | None = None,
    **kwargs,
) -> ad.AnnData:
    r"""Convert a protein group table stored in text format to :class:`anndata.AnnData`

    Parameters
    ----------
    file_path
        Path to text file of shape N (samples) x F (features). Expects observations (e.g. cells, samples) in rows
        and features (protein groups) in columns.
    row_type
        Whether features or observations are stored as rows. Per default, uses `row_type="features"`.
    separator
        File separator. Defaults to a tab-separated file.
    sample_metadata_index
        Indices of metadata for samples (either names or integer indices). If the `row_type` is "features" uses columns, if the `row_type` is "observations" uses rows.
        Defaults to the first index.
    feature_metadata_index
        Indices of metadata for features (either names or integer indices). If the `row_type` is "features" uses rows, if the `row_type` is "observations" uses columns
        Defaults to the first column.
    sample_name
        Column names for samples in `anndata.AnnData.obs` attribute. Must be the same length as `sample_metadata_index`.
        Defaults to the original name of the sample metadata.
    feature_name
        Column names fors features in `anndata.AnnData.var` attribute. Must be the same length as `feature_metadata_index`.
        Defaults to the original names of the feature metadata.
    **kwargs
        Keyword arguments passed to :func:`pandas.read_csv`

    Returns
    -------
    :class:`anndata.AnnData`
        AnnData object with N observations and F features.

            - .obs Contains content of the rows/columns indicated in sample_metadata_index
            - .var contains content of the rows/columns indicated in sa_metadata_index

    Example
    -------

    Here we showcase the default behaviour of the reader. A (features x observations) protein group matrix
    is converted into an :class:`anndata.AnnData` object.

    .. code-block:: python

        import pandas as pd
        import alphatools as at

        pg_matrix_path = ...

        # Inspect data
        pd.read_csv(pg_matrix_path, sep="\t", index=0).head(3)
        >            1   2   3  ...
           proteins
           P1        0   1   2  ...
           P2        3   4   5  ...
           P3        6   7   8  ...

        adata = at.io.read_pg_matrix(pg_matrix_path, sample_name = "sample_id")
        adata
        > AnnData object with n_obs x n_vars = 3 x 3
          obs: 'sample_id'
          var: 'proteins'
        adata.obs
        >       sample_id
            A   A
            B   B
            C   C
        adata.var
        >    proteins
         P1  'P1'
         P2  'P2'
         P3  'P3'

    If the observations are in the rows, you achieve the same behaviour by passing `row_type="observations"`

    .. code-block:: python
        # Inspect data
        pd.read_csv(pg_matrix_path, sep="\t", index=0).head(3)
        >           P1  P2  P3 ...
           samples
           1        0   1   2  ...
           2        3   4   5  ...
           3        6   7   8  ...

        adata = at.io.read_pg_matrix(pg_matrix_path, row_type="observations", feature_name="proteins")

    If the features or observations contain additional metadata, pass the column/row names as
    `feature_metadata_index` or `sample_metadata_index` to the function to add them to the `.var`
    respectively `.obs` attributes.

    .. code-block:: python

        # Inspect data
        pd.read_csv(pg_matrix_path, sep="\t", index=0).head(3)
        >                     1   2   3  ...
           protein seq
           P1       'AAA...'    0   1   2  ...
           P2       'CCC...'    3   4   5  ...
           P3       'DDD...'    6   7   8  ...

        adata = at.io.read_pg_matrix(pg_matrix_path, row_type="features", feature_metadata_index=["protein", "seq"])
        adata
        > AnnData object with n_obs x n_vars = 3 x 3
          obs: 'sample_id'
          var: 'protein' 'seq'
        adata.obs
        >       sample_id
            A   A
            B   B
            C   C
        adata.var
        >    protein seq
         P1  'P1' 'AAA'
         P2  'P2' 'CCC'
         P3  'P3' 'DDD'

    """
    _validate_row_type(row_type=row_type)

    if isinstance(sample_metadata_index, str | int):
        sample_metadata_index = [sample_metadata_index]
    if isinstance(feature_metadata_index, str | int):
        feature_metadata_index = [feature_metadata_index]

    if isinstance(sample_name, str):
        sample_name = [sample_name]
    if isinstance(feature_name, str):
        feature_name = [feature_name]

    # Map metadata indices depending on orientation of dataframe
    # to keep the API focussed on feature/sample syntax
    if row_type == "features":
        index_col = feature_metadata_index
        header = sample_metadata_index
    else:
        index_col = sample_metadata_index
        header = feature_metadata_index

    df = pd.read_csv(file_path, sep=separator, index_col=index_col, header=header, **kwargs)

    # Expects (observations x features)
    if row_type == "features":
        df = df.T

    if sample_name is not None:
        df = df.rename_axis(index=sample_name)

    if feature_name is not None:
        df = df.rename_axis(columns=feature_name)

    # X needs to be an array
    X = df.to_numpy()

    # Whether to set sample names and feature names as indices in anndata
    # Rename axis to prevent redundant naming with column names
    obs = df.index.to_frame(index=False).rename_axis(index=None)
    var = df.columns.to_frame(index=False).rename_axis(index=None)

    return ad.AnnData(X=X, obs=obs, var=var)
