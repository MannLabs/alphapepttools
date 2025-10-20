"""Factory class to convert PSM DataFrames to AnnData format."""

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from alphabase.psm_reader import PSMReaderBase
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.psm_reader import psm_reader_provider

from alphatools.pp.data import add_metadata


class AnnDataFactory:
    """Factory class to convert AlphaBase PSM DataFrames to AnnData format."""

    def __init__(
        self,
        psm_df: pd.DataFrame,
        intensity: str = PsmDfCols.INTENSITY,
        sample_id: str = PsmDfCols.RAW_NAME,
        feature_id: str = PsmDfCols.PROTEINS,
    ):
        """Initialize AnnDataFactory.

        The anndata factory retains its choice for intensity,
        sample_id and feature_id columns. This is a way to process
        any dataframe with the AnnDataFactory class, however the default
        use-case is to call it in the context of the psm_reader function,
        operating on alphabase-standardized column names (handled by from_files).

        Parameters
        ----------
        psm_df: pd.DataFrame
            Dataframe containing precursor intensity, sample_id and feature_id columns in a longtable
        intensity: str
            Column containing the precursor intensities
        sample_id: str
            Column containing the sample identifiers
        feature_id: str
            Column dictating which feature ends up as the AnnData's var_names after the pivoting operation

        """
        self._psm_df = psm_df
        self.intensity = intensity
        self.sample_id = sample_id
        self.feature_id = feature_id

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
            index=self.sample_id,
            columns=self.feature_id,
            values=self.intensity,
            aggfunc="first",  # DataFrameGroupBy.first -> will skip NA
            fill_value=np.nan,
            dropna=False,
        )

        # Create Nxp AnnData object where N=raw names and p=features (e.g. proteins)
        adata = ad.AnnData(
            X=pivot_df.values,
            obs=pd.DataFrame(index=pivot_df.index),
            var=pd.DataFrame(index=pivot_df.columns),
        )

        # Extract additional metadata if needed
        adata = self._add_metadata_from_columns(adata, var_columns, self.feature_id, axis=1)
        return self._add_metadata_from_columns(adata, obs_columns, self.sample_id, axis=0)

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

    @classmethod
    def from_files(
        cls,
        file_paths: str | list[str],
        reader_type: str = "maxquant",
        *,
        intensity_column: str | None = None,
        feature_id_column: str | None = None,
        sample_id_column: str | None = None,
        **kwargs,
    ) -> "AnnDataFactory":
        """Create AnnDataFactory from PSM files.

        Parameters
        ----------
        file_paths : Union[str, List[str]]
            Path(s) to PSM file(s)
        reader_type : str, optional
            Type of PSM reader to use, by default "maxquant"
        intensity_column: str, optional
            Name of the column storing intensity data. Default is taken from `psm_reader.yaml`
        protein_id_column: str, optional
            Name of the column storing proteins ids. Default is taken from `psm_reader.yaml`
        raw_name_column: str, optional
            Name of the column storing raw (or run) name. Default is taken from `psm_reader.yaml`
        **kwargs
            Additional arguments passed to PSM reader

        Returns
        -------
        AnnDataFactory
            Initialized AnnDataFactory instance

        """
        reader_config = cls._get_reader_configuration(reader_type)

        reader: PSMReaderBase = psm_reader_provider.get_reader(reader_type, **reader_config, **kwargs)

        custom_column_mapping = {
            k: v
            for k, v in {
                PsmDfCols.INTENSITY: intensity_column if intensity_column else None,
                PsmDfCols.PROTEINS: feature_id_column if feature_id_column else None,
                PsmDfCols.RAW_NAME: sample_id_column if sample_id_column else None,
            }.items()
            if v is not None
        }

        if custom_column_mapping:
            reader.add_column_mapping(custom_column_mapping)

        psm_df = reader.load(file_paths)
        return cls(psm_df)

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
