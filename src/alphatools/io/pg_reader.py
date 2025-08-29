from typing import Any

import anndata as ad
from alphabase.pg_reader.pg_reader import pg_reader_provider

SAMPLE_ID_NAME: str = "sample_id"


def read_pg_table(
    path: str,
    search_engine: str,
    *,
    column_mapping: dict[str, Any] | None = None,
    measurement_regex: str | None = None,
) -> ad.AnnData:
    """Read protein group table to the :class:`anndata.AnnData` format

    Read (features x observations) protein group matrices from proteomics search engines into
    the :class:`anndata.AnnData` format (observations x features). Per default,
    raw intensities are returned, which can be modified dependening on the search engine.

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
        Name of engine output, pass the method name of the corresponding reader. You can
        list all available readers with the :func:`dvpio.read.omics.available_reader` helper function
    column_mapping
        Mapping of additional columns in protein group table to a unified name, defaults to standard colum mapping in alphabase.
        Passed to :meth:`alphabase.pg_reader.pg_reader_provider.get_reader`.
        Expected format is

        .. code-block:: python

            {"new_column_name": "column_name_pg_matrix", ...}

    measurement_regex
        Regular expression that subsets feature columns to the correct quantification type. Only relevant if PG matrix contains multiple
        quantification methods per sample. Defaults to raw intensities. Options depend on the reader

            - None (default): Raw intensities
            - Reader-specific pre-configured names (e.g. `lfq`): Available intensities in the report (e.g. LFQ)
            - A valid regular expression

        Use classmethod `get_preconfigured_regex` for the respective reader in `alphabase`

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
    reader = pg_reader_provider.get_reader(
        search_engine, column_mapping=column_mapping, measurement_regex=measurement_regex
    )
    # Features x Observations
    df = reader.import_file(path)

    # Observations x Features
    return ad.AnnData(
        X=df.to_numpy().T, var=df.index.to_frame(index=False), obs=df.columns.to_frame(index=False, name=SAMPLE_ID_NAME)
    )
