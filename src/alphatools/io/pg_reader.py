from typing import Any

import anndata as ad
import pandas as pd
from alphabase.pg_reader.pg_reader import pg_reader_provider

SAMPLE_ID_NAME: str = "sample_id"


def read_pg_table(
    path: str,
    search_engine: str,
    *,
    column_mapping: dict[str, Any] | None = None,
    measurement_regex: str | None = None,
    **reader_provider_kwargs,
) -> ad.AnnData:
    """Read protein group table to the :class:`anndata.AnnData` format

    Read (features x observations) protein group matrices from proteomics search engines into
    the :class:`anndata.AnnData` format (observations x features). Per default,
    raw intensities are returned, which can be modified dependening on the search engine.
    If a single unique feature index could be derived from the input, the function
    will assign it as var index. Otherwise, an ascending integer var index will be used.

    Supported formats include

        - AlphaDIA (`alphadia`)
        - AlphaPept (`alphapept`, csv+hdf)
        - DIANN (`diann`)
        - MaxQuant (`maxquant`)
        - Spectronaut (`spectronaut`, parquet + tsv)

    See `alphabase.pg_reader` module for more information

    Parameters
    ----------
    path
        Path to protein group matrix
    search_engine
        Name of engine output, pass the method name of the corresponding reader.
    column_mapping
        Passed to :meth:`alphabase.pg_reader.pg_reader_provider.get_reader`.
        A dictionary of mapping alphabase columns (keys) to the corresponding columns in the other
        search engine (values). If `None` will be loaded from the `column_mapping` key of the respective
        search engine in `pg_reader.yaml`.
    measurement_regex
        Passed to :meth:`alphabase.pg_reader.pg_reader_provider.get_reader`.
        Regular expression that identifies correct measurement type. Only relevant if PG matrix contains multiple
        measurement types. For example, alphapept returns the raw protein intensity per sample in column `A` and the
        LFQ corrected value in `A_LFQ`. If `None` loads raw intensities.
    reader_provider_kwargs
        Passed to :meth:`alphabase.pg_reader.pg_reader_provider.get_reader`

    Returns
    -------
    :class:`anndata.AnnData`
        AnnData object that can be further processed with scVerse packages.

        - adata.X
            Stores values of the intensity columns in the report of shape observations x features
        - adata.obs
            Stores observations with protein group matrix sample names as `sample_id` column.
        - adata.var
            Stores features and feature metadata.

    Example
    -------

    .. code-block:: python

        from alphatools.io import read_pg_table

        alphadia_path = ...
        adata = read_pg_table(alphadia_path, search_engine="alphadia")

        maxquant_path = ...
        # Read LFQ values from MaxQuant report
        adata = read_pg_table(maxquant_path, search_engine="maxquant", measurement_regex="lfq")

    Get available regular expressions

    .. code-block:: python

        from alphabase.pg_reader import pg_reader_provider

        alphapept_reader = pg_reader_provider.get_reader("alphapept")
        alphapept_reader.get_preconfigured_regex()
        > {'raw': '^.*(?<!_LFQ)$', 'lfq': '_LFQ$'}

    See Also
    --------
    :mod:`alphabase.pg_reader`
    """
    # Build reader_provider_kwargs
    # This assures that the default values of the readers are considered (e.g. if `column_mapping="raw"`)
    if column_mapping is not None:
        reader_provider_kwargs["column_mapping"] = column_mapping
    if measurement_regex is not None:
        reader_provider_kwargs["measurement_regex"] = measurement_regex

    reader = pg_reader_provider.get_reader(search_engine, **reader_provider_kwargs)

    # Features x Observations
    df = reader.import_file(path)

    # Feature index logic: If first level is unique, use as var.index (with remaining levels as columns if multi-index), otherwise convert all to columns with integer index
    if df.index.get_level_values(0).is_unique:
        var_df = df.index.to_frame(index=True).iloc[:, 1:] if df.index.nlevels > 1 else pd.DataFrame(index=df.index)
    else:
        var_df = df.index.to_frame(index=False)

    # Observations x Features
    return ad.AnnData(X=df.to_numpy().T, var=var_df, obs=pd.DataFrame(index=df.columns))
