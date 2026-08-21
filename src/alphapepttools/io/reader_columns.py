"""Default column mappings for different search engines and data levels."""

from alphabase.psm_reader.keys import PsmDfCols

# Define for each feature level (proteins, precursors, peptides, genes)
# Which feature id and which intensity value should be used
# These columns might not be defined for some search engines.

# The feature id column is standardized across readers
ALPHAPEPTTOOLS_FEATURE_ID_NAME = "feature_id_column"

# The order of this dict is important for link construction between levels and proceeds from highest to lowest level, i.e. genes - proteins - peptides - precursors.
FEATURE_LEVEL_CONFIG = {
    "genes": {
        "intensity_column": PsmDfCols.GENE_INTENSITY,
        ALPHAPEPTTOOLS_FEATURE_ID_NAME: PsmDfCols.GENES,
        "sample_id_column": PsmDfCols.RAW_NAME,
    },
    "proteins": {
        "intensity_column": PsmDfCols.INTENSITY,
        ALPHAPEPTTOOLS_FEATURE_ID_NAME: PsmDfCols.PROTEINS,
        "sample_id_column": PsmDfCols.RAW_NAME,
    },
    "peptides": {
        "intensity_column": PsmDfCols.PEPTIDE_INTENSITY,
        ALPHAPEPTTOOLS_FEATURE_ID_NAME: PsmDfCols.SEQUENCE,
        "sample_id_column": PsmDfCols.RAW_NAME,
    },
    "precursors": {
        "intensity_column": PsmDfCols.PRECURSOR_INTENSITY,
        ALPHAPEPTTOOLS_FEATURE_ID_NAME: PsmDfCols.PRECURSOR_ID,
        "sample_id_column": PsmDfCols.RAW_NAME,
    },
}
