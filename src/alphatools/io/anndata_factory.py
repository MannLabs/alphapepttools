"""Factory class to convert PSM DataFrames to AnnData format."""

from types import MethodType
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

    def __init__(self, psm_df: pd.DataFrame):
        """Initialize AnnDataFactory.

        Parameters
        ----------
        psm_df : pd.DataFrame
            AlphaBase PSM DataFrame containing at minimum the columns:
            - PsmDfCols.RAW_NAME
            - PsmDfCols.PROTEINS
            - PsmDfCols.INTENSITY

        """
        self._psm_df = psm_df

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
            index=PsmDfCols.RAW_NAME,
            columns=PsmDfCols.PROTEINS,
            values=PsmDfCols.INTENSITY,
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
        adata = self._add_metadata_from_columns(adata, var_columns, PsmDfCols.PROTEINS, axis=1)
        return self._add_metadata_from_columns(adata, obs_columns, PsmDfCols.RAW_NAME, axis=0)

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

        columns = [columns] if isinstance(columns, str) else columns

        if index_column not in columns:
            columns.append(index_column)

        metadata = self._psm_df[columns].drop_duplicates().set_index(index_column, drop=True)

        return add_metadata(
            adata=adata,
            incoming_metadata=metadata,
            axis=axis,
            keep_data_shape=True,
            verbose=False,
        )

    @classmethod
    def from_df(
        cls,
        psm_df: pd.DataFrame,
        reader_type: str = "maxquant",
        *,
        intensity_column: str | None = None,
        feature_id_column: str | None = None,
        sample_id_column: str | None = None,
        **kwargs,
    ) -> "AnnDataFactory":
        """Create AnnDataFactory from a PSM DataFrame.

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM DataFrame with custom or standardized column names
        reader_type : str
            Kind of PSM reader, e.g. "alphadia", "diann", "spectronaut", "maxquant", etc.
        intensity_column : str, optional
            Name of the column storing intensity data. If not specified, assumes PsmDfCols.INTENSITY
        feature_id_column : str, optional
            Name of the column storing feature IDs. If not specified, assumes PsmDfCols.PROTEINS
        sample_id_column : str, optional
            Name of the column storing sample IDs. If not specified, assumes PsmDfCols.RAW_NAME

        Returns
        -------
        AnnDataFactory
            Initialized AnnDataFactory instance

        """
        # Create a copy to avoid modifying the original DataFrame
        psm_df = psm_df.copy()

        reader_config = cls._get_reader_configuration(reader_type)

        # column_mapping here??
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

        # Helper function to apply alphabase standardization to dataframe
        def _load_file_from_df(self, filename) -> pd.DataFrame:  # noqa: ANN001
            return psm_df

        reader._load_file = MethodType(_load_file_from_df, reader)  # noqa: SLF001
        psm_df = reader.import_file("dummy_path")

        return cls(psm_df)

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
