import anndata as ad

from .anndata_factory import AnnDataFactory


def read_psm_table(
    file_paths: str | list[str],
    search_engine: str,
    level: str = "proteins",
    *,
    intensity_column: str | None = None,
    feature_id_column: str | None = None,
    sample_id_column: str | None = None,
    var_columns: str | list[str] | None = None,
    obs_columns: str | list[str] | None = None,
    **kwargs,
) -> ad.AnnData:
    """Read peptide spectrum match tables to the :class:`anndata.AnnData` format

    Read peptide spectrum match (PSM) tables from proteomics search engines into
    the :class:`anndata.AnnData` format (observations x features). Per default,
    raw protein intensities are returned.

    Supported formats include

        - AlphaDIA (`alphadia`)
        - AlphaPept (`alphapept`)
        - DIANN (`diann`)
        - MaxQuant (`maxquant`)
        - Spectronaut (`spectronaut`, parquet + tsv)

    Parameters
    ----------
    file_paths
        Path to peptide spectrum match reports. If a list of reports is passed, all must be from the same search engine.
    search_engine
        Name of search engine that generated the output, pass the method name of the corresponding reader.
    level
        Level of quantification to read. One of "proteins", "precursors", or "genes". Defaults to "proteins".
    intensity_column
        Column that holds the quantified intensities in the PSM table. Defaults to the pre-configured protein intensities value
        in `alphabase`.
    feature_id_column
        Column that holds the feature identifier in the PSM table. Defaults to proteins and the pre-configured value
        in `alphabase`.
    sample_id_column
        Column that holds the sample identifier in the PSM table. Defaults to the pre-configured value
        in `alphabase`.
    var_columns
        Additional columns to annotate features in the `adata.var` table. Can be a single column name or a list of column names.
        Defaults to None.
    obs_columns
        Additional columns to annotate observations in the `adata.obs` table. Can be a single column name or a list of column names.
        Defaults to None.
    **kwargs
        Keyword arguments passed to :meth:`alphabase.psm_reader.psm_reader_provider.get_reader`

    Returns
    -------
    :class:`anndata.AnnData`
        AnnData object that can be further processed with scVerse packages.

        - adata.X
            Stores values of the intensity columns in the report of shape observations x features
        - adata.obs
            Stores observations with protein group matrix sample names as `sample_id` column.
        - adata.var
            Stores features and feature metadata with standardized alphabase names.

    Example
    -------

    .. code-block:: python

        import alphatools as at

        alphadia_path = ...
        adata = at.io.read_psm_table(alhpadia_path, search_engine="alphadia")


    See Also
    --------
    :mod:`alphabase.psm_reader`

    """
    return AnnDataFactory.from_files(
        file_paths=file_paths,
        reader_type=search_engine,
        level=level,
        intensity_column=intensity_column,
        feature_id_column=feature_id_column,
        sample_id_column=sample_id_column,
        **kwargs,
    ).create_anndata(
        var_columns=var_columns,
        obs_columns=obs_columns,
    )
