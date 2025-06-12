from typing import Literal

import anndata as ad
import pandas as pd


def read_pg_matrix(
    file_path: str,
    sample_name: str = "sample_id",
    feature_name: str = "pg",
    *,
    set_index: bool = True,
    row_type: Literal["observation", "features"] = "features",
    separator: str = "\t",
    index_col: str | int = 0,
    header: str | int = 0,
    **kwargs,
) -> ad.AnnData:
    r"""Convert a protein group table stored in text format to :class:`anndata.AnnData`

    Parameters
    ----------
    file_path
        Path to text file of shape N (samples) x F (features). Expects observations (e.g. cells, samples) in rows
        and features (protein groups) in columns
    sample_name
        Column name of samples in `anndata.AnnData.obs` attribute (defaults to `sample_id`)
    feature_name
        Column name of features in `anndata.AnnData.var` attribute (defaults to `pg`)
    set_index
        Whether to set the sample names and feature names as indices in the :class:`anndata.AnnData` object in addition to the respective columns
        If False uses default initialization with numeric indices.
    row_type
        Whether features or observations are stored as rows. Per default, uses `row_type="features"`.
    separator
        File separator. Defaults to a tab-separated file
    index_col
        Index column. Defaults to first column in protein group table
    header
        Indexing row. Defaults to first row in protein group table
    **kwargs
        Keyword arguments passed to :func:`pandas.read_csv`

    Returns
    -------
    :class:`anndata.AnnData`
        AnnData object with N observations and F features.

            - .obs Contains content of df.index
            - .var contains content of df.columns

    Example
    -------

    Here we showcase the default behaviour of the reader. A (features x observations) protein group matrix
    gets converted to an :class:`anndata.AnnData` object.

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

        adata = at.io.read_pg_matrix(pg_matrix_path)
        adata
        > AnnData object with n_obs x n_vars = 3 x 3
          obs: 'sample_id'
          var: 'pg'
        adata.obs
        >       sample_id
            A   A
            B   B
            C   C
        adata.var
        >    pg
         G1  G1
         G2  G2
         G3  G3

    If the observations are in the rows, you achieve the same behaviour by passing `row_type="observations"`

    .. code-block:: python
        # Inspect data
        pd.read_csv(pg_matrix_path, sep="\t", index=0).head(3)
        >           P1  P2  P3 ...
           samples
           1        0   1   2  ...
           2        3   4   5  ...
           3        6   7   8  ...

        adata = at.io.read_pg_matrix(pg_matrix_path, row_type="observations")
    """
    df = pd.read_csv(file_path, sep=separator, index_col=index_col, header=header, **kwargs)

    # Expects (observations x features)
    if row_type == "features":
        df = df.T

    df = df.rename_axis(index=sample_name, columns=feature_name)

    X = df.to_numpy()
    # Whether to set sample names and feature names as indices in anndata
    # Rename axis to prevent redundant naming with column names
    obs = df.index.to_frame(index=set_index).rename_axis(index=None)
    var = df.columns.to_frame(index=set_index).rename_axis(index=None)

    return ad.AnnData(X=X, obs=obs, var=var)
