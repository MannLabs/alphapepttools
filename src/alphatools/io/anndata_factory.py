"""Factory class to convert PSM DataFrames to AnnData format."""

from typing import Any

import anndata as ad
import pandas as pd
from alphabase.psm_reader import PSMReaderBase
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.psm_reader import psm_reader_provider

from alphatools.io.reader_columns import DEFAULT_COLUMNS_DICT
from alphatools.pp.data import add_metadata


class AnnDataFactory:
    """Factory class to convert AlphaBase PSM DataFrames to AnnData format."""

    def __init__(
        self,
        psm_df: pd.DataFrame,
        intensity_column: str,
        sample_id_column: str,
        feature_id_column: str,
    ):
        """Initialize AnnDataFactory.

        This is a way to process any dataframe with the AnnDataFactory class,
        however the default use-case is to call it in the context of the psm_reader
        function, operating on alphabase-standardized column names (handled by from_files).

        Parameters
        ----------
        psm_df: pd.DataFrame
            Dataframe containing precursor intensity, sample_id and feature_id columns in a longtable
        intensity_column: str
            Column containing the precursor intensities
        sample_id_column: str
            Column containing the sample identifiers
        feature_id_column: str
            Column dictating which feature ends up as the AnnData's var_names after the pivoting operation

        """
        self._psm_df = psm_df
        self._intensity_column = intensity_column
        self._sample_id_column = sample_id_column
        self._feature_id_column = feature_id_column

    def _add_metadata_from_columns(
        self,
        adata: ad.AnnData,
        columns: str | list[str] | None,
        index_column: str,
        axis: int,
    ) -> ad.AnnData:
        """Add metadata to AnnData object from specified columns.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object to add metadata to
        columns : str | list[str] | None
            Columns to extract as metadata
        index_column : str
            Column to use as index
        axis : int
            0 for obs (samples), 1 for var (features)

        Returns
        -------
        ad.AnnData
            AnnData object with added metadata
        """
        if columns is None:
            return adata

        # Normalize to list and create a copy to avoid mutating caller's input
        columns = [columns] if isinstance(columns, str) else list(columns)

        if index_column not in columns:
            columns.append(index_column)

        # Set index first, then drop duplicates based on index only
        metadata = self._psm_df[columns].set_index(index_column, drop=True)
        metadata.index.name = None
        metadata = metadata[~metadata.index.duplicated(keep="first")]

        return add_metadata(
            adata=adata,
            incoming_metadata=metadata,
            axis=axis,
            keep_data_shape=True,
            verbose=False,
        )

    def create_anndata(
        self,
        var_columns: str | list[str] | None = None,
        obs_columns: str | list[str] | None = None,
    ) -> ad.AnnData:
        """Create AnnData object from PSM DataFrame.

        Parameters
        ----------
        var_columns : Union[str, List[str]], optional
            Additional columns to include in `var` of the AnnData object, by default None
        obs_columns : Union[str, List[str]], optional
            Additional columns to include in `obs` of the AnnData object, by default None

        Returns
        -------
        ad.AnnData
            AnnData object where:
            - obs (rows) are raw names
            - var (columns) are proteins
            - X contains intensity values

        """
        # Create pivot table: raw names x proteins with intensity values
        pivot_df = pd.pivot_table(
            self._psm_df,
            index=self._sample_id_column,
            columns=self._feature_id_column,
            values=self._intensity_column,
            aggfunc="first",  # DataFrameGroupBy.first -> will skip NA
            dropna=False,
        )

        # Create Nxp AnnData object where N=raw names and p=features (e.g. proteins)
        adata = ad.AnnData(
            X=pivot_df.values,
            obs=pd.DataFrame(index=pivot_df.index),
            var=pd.DataFrame(index=pivot_df.columns),
        )

        # Extract additional metadata if needed
        adata = self._add_metadata_from_columns(adata, var_columns, self._feature_id_column, axis=1)
        return self._add_metadata_from_columns(adata, obs_columns, self._sample_id_column, axis=0)

    @staticmethod
    def _identify_non_alphabase_columns(reader_type: str) -> list[str]:
        """Identify columns from READER_COLUMNS that are not covered by PsmDfCols.

        Parameters
        ----------
        reader_type : str
            Type of PSM reader

        Returns
        -------
        list[str]
            List of column names that need special retention (not in PsmDfCols)
        """
        # Get all required columns from all levels for this reader type from the defaults
        required_columns = list(
            {
                col_value
                for level_dict in DEFAULT_COLUMNS_DICT.get(reader_type, {}).values()
                for col_value in level_dict.values()
            }
        )

        # Get all PsmDfCols constant values (the actual column name strings)
        psm_df_cols_values = PsmDfCols.get_values()

        # Return those columns which are required for the current reader but not covered by PsmDfCols yet
        return [col for col in required_columns if col not in psm_df_cols_values]

    @classmethod
    def _get_reader_configuration(cls, reader_type: str) -> dict[str, dict[str, Any]]:
        """Get reader-specific configuration for mapping PSMs to anndata."""
        reader_configs = {
            "diann": {
                "filter_first_search_fdr": True,
                "filter_second_search_fdr": True,
            }
        }
        return reader_configs.get(reader_type, {})

    @classmethod
    def from_files(
        cls,
        file_paths: str | list[str],
        reader_type: str = "maxquant",
        level: str = "proteins",
        *,
        intensity_column: str | None = None,
        feature_id_column: str | None = None,
        sample_id_column: str | None = None,
        additional_columns: list[str] | None = None,
        **reader_kwargs,
    ) -> "AnnDataFactory":
        """Create AnnDataFactory from PSM files.

        Parameters
        ----------
        file_paths : Union[str, List[str]]
            Path(s) to PSM file(s)
        reader_type : str, optional
            Type of PSM reader to use, by default "maxquant"
        level : str, optional
            Level of quantification to read. One of "proteins", "precursors", or "genes". Defaults to "proteins".
        intensity_column: str, optional
            Name of the column storing intensity data. Default is taken from `psm_reader.yaml`
        feature_id_column: str, optional
            Name of the column storing feature ids. Default is taken from `psm_reader.yaml`
        sample_id_column: str, optional
            Name of the column storing sample ids. Default is taken from `psm_reader.yaml`
        additional_columns: list[str], optional
            Additional column names to be directly retained from the psm-table in order to enable experiment-specific
            metadata retention.
        **reader_kwargs
            Additional arguments passed to PSM reader

        Returns
        -------
        AnnDataFactory
            Initialized AnnDataFactory instance

        """
        reader_config = cls._get_reader_configuration(reader_type)

        reader: PSMReaderBase = psm_reader_provider.get_reader(reader_type, **reader_config, **reader_kwargs)

        # Identify the columns we need for this reader, but which are not yet covered by alphabase PsmDfCols
        # TODO: Once alphabase is updated we don't need this anymore since all columns will be covered by PsmDfCols
        extra_columns = cls._identify_non_alphabase_columns(reader_type)

        # Add user-specified additional columns which are not in PsmDfCols.get_values()
        if additional_columns:
            additional_columns = [col for col in additional_columns if col not in PsmDfCols.get_values()]
            extra_columns.extend(additional_columns)

        # Add identity mappings for extra columns so they're retained during reading
        if extra_columns:
            extra_column_mapping = {col: col for col in extra_columns}
            reader.add_column_mapping(extra_column_mapping)

        psm_df = reader.load(file_paths)

        # Get defaults for this reader/level, user input overrides
        defaults = DEFAULT_COLUMNS_DICT.get(reader_type, {}).get(level, {})
        intensity_column = intensity_column or defaults.get("intensity_column")
        feature_id_column = feature_id_column or defaults.get("feature_id_column")
        sample_id_column = sample_id_column or defaults.get("sample_id_column")

        # Validate that all required columns are present
        if intensity_column is None:
            msg = f"intensity_column is required but not provided and no default found for reader_type='{reader_type}' and level='{level}'"
            raise ValueError(msg)
        if feature_id_column is None:
            msg = f"feature_id_column is required but not provided and no default found for reader_type='{reader_type}' and level='{level}'"
            raise ValueError(msg)
        if sample_id_column is None:
            msg = f"sample_id_column is required but not provided and no default found for reader_type='{reader_type}' and level='{level}'"
            raise ValueError(msg)

        return cls(
            psm_df,
            intensity_column=intensity_column,
            feature_id_column=feature_id_column,
            sample_id_column=sample_id_column,
        )
