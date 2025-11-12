"""Default column mappings for different search engines and data levels."""

from alphabase.psm_reader.keys import PsmDfCols

# Define for each feature level (proteins, precursors, peptides, genes)
# Which feature id and which intensity value should be used
# These columns might not be defined for some search engines.
FEATURE_LEVEL_CONFIG = {
    "proteins": {
        "intensity_column": PsmDfCols.INTENSITY,
        "feature_id_column": PsmDfCols.PROTEINS,
        "sample_id_column": PsmDfCols.RAW_NAME,
    },
    "precursor": {
        "intensity_column": PsmDfCols.PRECURSOR_INTENSITY,
        "feature_id_column": PsmDfCols.PRECURSOR_ID,
        "sample_id_column": PsmDfCols.RAW_NAME,
    },
    "peptides": {
        "intensity_column": PsmDfCols.PEPTIDE_INTENSITY,
        "feature_id_column": PsmDfCols.SEQUENCE,
        "sample_id_column": PsmDfCols.RAW_NAME,
    },
    "genes": {
        "intensity_column": PsmDfCols.GENE_INTENSITY,
        "feature_id_column": PsmDfCols.GENES,
        "sample_id_column": PsmDfCols.RAW_NAME,
    },
}
