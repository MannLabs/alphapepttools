import anndata as ad

from .anndata_factory import AnnDataFactory


def read_psm_table(
    file_paths: str | list[str],
    search_engine: str,
    *,
    intensity_column: str | None = None,
    feature_id_column: str | None = None,
    sample_id_column: str | None = None,
    var_cols: str | list[str] | None = None,
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
    intensity_column
        Column that holds the quantified intensities in the PSM table. Defaults to the pre-configured protein intensities value
        in `alphabase`.
    feature_id_column
        Column that holds the feature identifier in the PSM table. Defaults to proteins and the pre-configured value
        in `alphabase`.
    sample_id_column
        Column that holds the sample identifier in the PSM table. Defaults to the pre-configured value
        in `alphabase`.
    var_cols
        Additional feature metadata columns that should be stored in final anndata object. Defaults to `None`.
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
        intensity_column=intensity_column,
        protein_id_column=feature_id_column,
        raw_name_column=sample_id_column,
        **kwargs,
    ).create_anndata(var_cols=var_cols)
