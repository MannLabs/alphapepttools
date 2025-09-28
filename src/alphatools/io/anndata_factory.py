"""Factory class to convert PSM DataFrames to AnnData format."""

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from alphabase.psm_reader import PSMReaderBase
from alphabase.psm_reader.keys import PsmDfCols

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
        required_cols = [PsmDfCols.RAW_NAME, PsmDfCols.PROTEINS, PsmDfCols.INTENSITY]
        missing_cols = [col for col in required_cols if col not in psm_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self._psm_df = psm_df

    def create_anndata(self, secondary_id_columns: str | list[str] | None = None) -> ad.AnnData:
        """Create AnnData object from PSM DataFrame.

        Parameters
        ----------
        secondary_id_columns : Union[str, List[str]], optional
            Additional columns to include in `var` of the AnnData object, by default None

        Returns
        -------
        ad.AnnData
            AnnData object where:
            - obs (rows) are raw names
            - var (columns) are proteins
            - X contains intensity values

        """
        # Extract additional feature metadata if needed
        if secondary_id_columns:
            if isinstance(secondary_id_columns, str):
                secondary_id_columns = [secondary_id_columns]
        else:
            secondary_id_columns = []

        # Ensure PsmDfCols.PROTEINS is included in feature metadata
        if PsmDfCols.PROTEINS not in secondary_id_columns:
            secondary_id_columns.append(PsmDfCols.PROTEINS)

        feature_metadata = self._psm_df[secondary_id_columns].drop_duplicates().set_index(PsmDfCols.PROTEINS, drop=True)

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

        adata = ad.AnnData(
            X=pivot_df.values,
            obs=pd.DataFrame(index=pivot_df.index),
            var=pd.DataFrame(index=pivot_df.columns),
        )

        # Add feature metadata to var of the resulting AnnData
        return add_metadata(
            adata=adata,
            incoming_metadata=feature_metadata,
            axis=1,
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
        from alphabase.psm_reader.psm_reader import psm_reader_provider

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
