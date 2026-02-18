import anndata as ad
import pandas as pd

import alphapepttools as apt


def _create_peptide_group_id(sequence: pd.Series, protein_id: pd.Series, separator: str = "_IN_") -> pd.Series:
    """Create unique identifier by concatenating sequence and protein ID.

    Parameters
    ----------
    sequence : pd.Series
        Peptide sequences
    protein_id : pd.Series
        Protein identifiers
    separator : str, optional
        Separator string between sequence and protein ID (default: "_IN_")

    Returns
    -------
    pd.Series
        Concatenated identifiers
    """
    return sequence.astype(str) + separator + protein_id.astype(str)


def _parse_peptide_group_id(group_id: pd.Index, separator: str = "_IN_") -> tuple[pd.Series, pd.Series]:
    """Parse peptide group ID back into sequence and protein ID components.

    Parameters
    ----------
    group_id : pd.Index
        Concatenated peptide group identifiers
    separator : str, optional
        Separator string used in concatenation (default: "_IN_")

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Tuple of (sequence, protein_id) Series
    """
    split = group_id.str.split(separator, n=1)
    sequence = split.str[0]
    protein_id = split.str[1]
    return sequence, protein_id


def _extract_quantification_data(adata: ad.AnnData, sequence_column: str, protein_id_column: str) -> pd.DataFrame:
    """Extract quantification matrix with sequence and protein metadata.

    Parameters
    ----------
    adata : ad.AnnData
        Input AnnData object with peptides/precursors as variables
    sequence_column : str
        Column name for peptide sequences
    protein_id_column : str
        Column name for protein identifiers

    Returns
    -------
    pd.DataFrame
        Quantification data with sequence and protein ID columns
    """
    # Transpose so peptides are rows
    adata_t = adata.T.copy()

    # Join quantification data with metadata
    metadata_columns = [sequence_column, protein_id_column]
    return adata_t.to_df().join(adata_t.obs[metadata_columns])


def _aggregate_by_sequence_and_protein(
    data_df: pd.DataFrame,
    sequence_column: str,
    protein_id_column: str,
    aggregation: str,
) -> pd.DataFrame:
    """Aggregate peptide data by sequence and protein group.

    Groups identical peptide sequences within the same protein group,
    aggregating quantification values and removing duplicates.

    Parameters
    ----------
    data_df : pd.DataFrame
        Quantification dataframe with sequence and protein ID columns
    sequence_column : str
        Column name for peptide sequences
    protein_id_column : str
        Column name for protein identifiers
    aggregation : str
        Aggregation method (e.g., 'max', 'mean', 'median')

    Returns
    -------
    pd.DataFrame
        Aggregated data with group ID as index
    """
    # Create composite grouping identifier
    group_id = _create_peptide_group_id(data_df[sequence_column], data_df[protein_id_column])
    data_df["_group_id"] = group_id

    # Drop metadata columns before aggregation
    quant_df = data_df.drop(columns=[sequence_column, protein_id_column])

    # Aggregate by group ID
    aggregated_df = quant_df.groupby("_group_id", observed=False).agg(aggregation)

    # Remove duplicates (same stripped sequence in protein group)
    aggregated_df = aggregated_df.drop_duplicates()

    # Parse group ID back into sequence and protein columns
    sequence, protein_id = _parse_peptide_group_id(aggregated_df.index)
    aggregated_df[sequence_column] = sequence
    aggregated_df[protein_id_column] = protein_id

    return aggregated_df


def _add_metadata_columns(
    aggregated_df: pd.DataFrame,
    adata_obs: pd.DataFrame,
    sequence_column: str,
    protein_id_column: str,
    added_columns: list[str],
) -> pd.DataFrame:
    """Merge additional metadata columns into aggregated dataframe.

    Parameters
    ----------
    aggregated_df : pd.DataFrame
        Aggregated quantification data
    adata_obs : pd.DataFrame
        Original AnnData.obs with metadata
    sequence_column : str
        Column name for peptide sequences
    protein_id_column : str
        Column name for protein identifiers
    added_columns : list[str]
        Additional metadata columns to include

    Returns
    -------
    pd.DataFrame
        Aggregated data with additional metadata merged
    """
    # remove sequence_column and protein_id_column from added_columns if they are present
    added_columns = [col for col in added_columns if col not in [sequence_column, protein_id_column]]

    # Prepare metadata to merge
    merge_columns = [*added_columns, sequence_column, protein_id_column]
    metadata_to_add = adata_obs[merge_columns].drop_duplicates()

    # Merge on sequence and protein ID
    merge_keys = [sequence_column, protein_id_column]
    merged_df = aggregated_df.merge(metadata_to_add, left_on=merge_keys, right_on=merge_keys, how="left")

    # Recreate index as composite ID
    merged_df.index = _create_peptide_group_id(merged_df[sequence_column], merged_df[protein_id_column])
    merged_df.index = merged_df.index.astype(str)

    return merged_df


def _create_anndata_from_aggregated(
    aggregated_df: pd.DataFrame,
    original_adata: ad.AnnData,
    sequence_column: str,
    protein_id_column: str,
    added_columns: list[str] | None,
) -> ad.AnnData:
    """Construct AnnData object from aggregated peptide data.

    Parameters
    ----------
    aggregated_df : pd.DataFrame
        Aggregated quantification data with metadata
    original_adata : ad.AnnData
        Original AnnData object
    sequence_column : str
        Column name for peptide sequences
    protein_id_column : str
        Column name for protein identifiers
    added_columns : list[str] | None
        Additional metadata columns included

    Returns
    -------
    ad.AnnData
        New AnnData object with aggregated peptide groups
    """
    # Determine variable metadata columns
    var_columns = [sequence_column, protein_id_column]
    if added_columns is not None:
        var_columns = pd.Series(added_columns + var_columns).unique().tolist()

    # Extract variable metadata
    var_metadata = aggregated_df.loc[:, var_columns]

    # Extract quantification matrix and transpose back to samples * peptides
    quant_matrix = aggregated_df.drop(columns=var_columns).T

    # Create AnnData object with basic index information
    return_adata = ad.AnnData(X=quant_matrix)

    # Use add_metadata to properly add sample metadata from original
    return_adata = apt.pp.add_metadata(return_adata, original_adata.obs, axis=0)

    # Use add_metadata to properly add peptide metadata
    return apt.pp.add_metadata(return_adata, var_metadata, axis=1)


def group_peptides(
    adata: ad.AnnData,
    sequence_column: str = "sequence",
    protein_id_column: str = "proteins",
    aggregation: str = "max",
    added_columns: list[str] | None = None,
) -> ad.AnnData:
    """Aggregate peptide sequences by sequence and protein group.

    Groups identical peptide sequences within the same protein group,
    aggregating quantification values using the specified method. This is
    useful for collapsing modified peptides or redundant identifications
    to unique sequence-protein combinations.

    The function:
    1. Creates composite identifiers from sequence + protein ID
    2. Aggregates quantification values for duplicate sequence-protein pairs
    3. Removes duplicates within protein groups
    4. Optionally merges additional metadata columns
    5. Returns a new AnnData object with aggregated data

    Parameters
    ----------
    adata : ad.AnnData
        Input AnnData with peptides/precursors as variables (columns).
        Must contain `sequence_column` and `protein_id_column` in `.var`.
    sequence_column : str, optional
        Column name in `.var` containing peptide sequences (default: "sequence")
    protein_id_column : str, optional
        Column name in `.var` containing protein identifiers (default: "proteins")
    aggregation : str, optional
        Aggregation method for duplicate entries. Common options: "max", "mean",
        "median", "sum" (default: "max")
    added_columns : list[str] | None, optional
        Additional metadata columns from `.var` to include in output (default: None)

    Returns
    -------
    ad.AnnData
        New AnnData object with:
        - `.X`: Aggregated quantification matrix (samples * peptide groups)
        - `.obs`: Sample metadata from original (unchanged)
        - `.var`: Peptide group metadata with sequence, protein ID, and any
          additional columns specified

    Raises
    ------
    TypeError
        If `added_columns` is not a list

    Examples
    --------
    Basic usage with default parameters:

    .. code-block:: python

        import anndata as ad
        from alphasite.sequence_tools import group_peptides

        # Load precursor data
        adata = ad.read_h5ad("precursors.h5ad")

        # Aggregate to unique sequence-protein combinations
        adata_peptides = group_peptides(adata)

    Include additional metadata:

    .. code-block:: python

        # Include gene names in output
        adata_peptides = group_peptides(adata, added_columns=["genes", "kinase_status"])

    Use mean aggregation instead of max:

    .. code-block:: python

        adata_peptides = group_peptides(adata, aggregation="mean")

    Notes
    -----
    - The function transposes the input AnnData internally (peptides as rows)
      for processing, then transposes back for output
    - Composite identifiers use format: "{sequence}_IN_{protein_id}"
    - Duplicates are removed after aggregation (same stripped sequence in
      protein group)
    - Missing values in metadata columns will be NaN after merge
    """
    # Validate input
    if added_columns is not None and not isinstance(added_columns, list):
        raise TypeError("added_columns must be a list")

    # Step 1: Extract quantification data with metadata
    data_df = _extract_quantification_data(adata, sequence_column, protein_id_column)

    # Step 2: Aggregate by sequence and protein
    aggregated_df = _aggregate_by_sequence_and_protein(data_df, sequence_column, protein_id_column, aggregation)

    # Step 3: Add additional metadata if requested
    if added_columns is not None:
        adata_t = adata.T.copy()
        aggregated_df = _add_metadata_columns(
            aggregated_df,
            adata_t.obs,
            sequence_column,
            protein_id_column,
            added_columns,
        )

    # Step 4: Create output AnnData object
    return _create_anndata_from_aggregated(
        aggregated_df,
        adata,
        sequence_column,
        protein_id_column,
        added_columns,
    )
