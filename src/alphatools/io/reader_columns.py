"""Default column mappings for different search engines and data levels."""

from alphabase.psm_reader.keys import PsmDfCols

DEFAULT_COLUMNS_DICT = {
    "diann": {
        "proteins": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
        "precursors": {
            "intensity_column": "Precursor.Normalised",
            "feature_id_column": "Precursor.Id",
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
        "genes": {
            "intensity_column": "Genes.MaxLFQ",
            "feature_id_column": PsmDfCols.GENES,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
    },
    "alphadia": {
        "proteins": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
        "precursors": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
        "genes": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
    },
    "alphapept": {
        "proteins": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
        "precursors": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
        "genes": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
    },
    "maxquant": {
        "proteins": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
        "precursors": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
        "genes": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
    },
    "spectronaut": {
        "proteins": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
        "precursors": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
        "genes": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
    },
    "sage": {
        "proteins": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
        "precursors": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
        "genes": {
            "intensity_column": PsmDfCols.INTENSITY,
            "feature_id_column": PsmDfCols.PROTEINS,
            "sample_id_column": PsmDfCols.RAW_NAME,
        },
    },
}
