from typing import Literal

import anndata as ad

from alphapepttools.io.reader_columns import FEATURE_LEVEL_CONFIG

from .anndata_factory import AnnDataFactory


def read_psm_table(
    file_paths: str | list[str],
    search_engine: str,
    level: Literal["proteins", "genes", "peptides", "precursors"] = "proteins",
    *,
    intensity_column: str | None = None,
    feature_id_column: str | None = None,
    sample_id_column: str | None = None,
    var_columns: str | list[str] | None = None,
    obs_columns: str | list[str] | None = None,
    **reader_kwargs,
) -> ad.AnnData:
    """Read peptide spectrum match tables to the :class:`anndata.AnnData` format

    Read peptide spectrum match (PSM) tables from proteomics search engines into
    the :class:`anndata.AnnData` format (observations x features). Per default,
    raw protein intensities are returned. Additionally, custom columns can be selected
    to be retained in the resulting AnnData object.

    Note: The underlying pivoting function will aggregate metadata in a "first" manner,
    meaning that if the metadata is finer grained than the feature level, information will
    be lost. An example for this is setting feature_id_column="protein_ids" and setting
    "var_columns" to include peptide sequences. This produces a protein-level AnnData
    object with one peptide sequence per protein, which is likely not desired. Therefore,
    ensure that the metadata you want to retain is actually applicable to the feature level.

    Supported formats include

        - AlphaDIA (`alphadia`)
        - AlphaPept (`alphapept`)
        - DIANN (`diann`)
        - MaxQuant (`maxquant`)
        - Spectronaut (`spectronaut`, parquet + tsv)

    Get supported search engines with `alphapepttools.io.list_available_reader(kind="psm_reader")`

    Parameters
    ----------
    file_paths
        Path to peptide spectrum match reports. If a list of reports is passed, all must be from the same search engine.
    search_engine
        Name of search engine that generated the output.
    level
        Level of quantification to read. One of
            - `proteins`
            - `precursors`
            - `genes`
            - `peptides`
    intensity_column
        Column that holds the quantified intensities in the PSM table. If `None`, defaults to the pre-configured intensity column
        for the specified level.
    feature_id_column
        Column that holds the feature identifier in the PSM table. If `None`, defaults to the pre-configured feature identifier column
        for the specified level.
    sample_id_column
        Column that holds the sample identifier in the PSM table. If `None`, defaults to the pre-configured sample identifier column
        for the specified level.
    var_columns
        Additional columns to annotate features in the `adata.var` table. Can be a single column name or a list of column names.
        Defaults to None.
    obs_columns
        Additional columns to annotate observations in the `adata.obs` table. Can be a single column name or a list of column names.
        Defaults to None.
    **reader_kwargs
        Keyword arguments passed to :meth:`alphabase.psm_reader.psm_reader_provider.get_reader`

    Returns
    -------
    :class:`anndata.AnnData`
        AnnData object that can be further processed with scVerse packages.

        - adata.X
            Stores values of the intensity columns in the report of shape observations x features.
        - adata.obs
            Stores observations with protein group matrix sample names as `sample_id` column and additional `obs_columns`.
        - adata.var
            Stores features and feature metadata with standardized alphabase names and additional `var_columns`.

    Example
    -------

    .. code-block:: python

        import alphapepttools as at

        # Read PSM report (defaults to protein level)
        adata_proteins = at.io.read_psm_table(alphadia_path, search_engine="alphadia")

        # Read precursor intensities from PSM report
        adata_precursors = at.io.read_psm_table(alphadia_path, search_engine="alphadia", level="precursors")

        # Read a non-default intensity column
        adata = at.io.read_psm_table(
            diann_path, search_engine="alphadia", level="precursors", intensity_column="Precursors.Quantity"
        )


    See Also
    --------
    :mod:`alphabase.psm_reader`
    :func:`alphapepttools.io.list_available_reader`

    """
    # Determine which data & metadata columns are requested
    requested_columns = [col for col in [intensity_column, feature_id_column, sample_id_column] if col is not None]

    for col in [var_columns, obs_columns]:
        if col is not None:
            if isinstance(col, list):
                requested_columns.extend(col)
            else:
                requested_columns.append(col)

    # In case users request columns by their alphabase psm reader name, we skip looking for them in the original dataset
    alphabase_reader_columns = set()
    for alphabase_reader_column in FEATURE_LEVEL_CONFIG:
        alphabase_reader_columns.update(FEATURE_LEVEL_CONFIG[alphabase_reader_column].values())
    additional_columns = [col for col in requested_columns if col not in alphabase_reader_columns]

    if not additional_columns:
        additional_columns = None

    return AnnDataFactory.from_files(
        file_paths=file_paths,
        reader_type=search_engine,
        additional_columns=additional_columns,
        **reader_kwargs,
    ).create_anndata(
        level=level,
        intensity_column=intensity_column,
        feature_id_column=feature_id_column,
        sample_id_column=sample_id_column,
        var_columns=var_columns,
        obs_columns=obs_columns,
    )
