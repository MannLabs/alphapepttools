"""Factory class to convert PSM DataFrames to AnnData format."""

from typing import Any

import anndata as ad
import pandas as pd
from alphabase.psm_reader import PSMReaderBase
from alphabase.psm_reader.psm_reader import psm_reader_provider

from alphapepttools.io.reader_columns import FEATURE_LEVEL_CONFIG
from alphapepttools.pp.data import add_metadata


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
        psm_df
            Dataframe containing precursor intensity, sample_id and feature_id columns in a longtable
        intensity_column
            Column containing the precursor intensities
        sample_id_column
            Column containing the sample identifiers
        feature_id_column
            Column dictating which feature ends up as the AnnData's var_names after the pivoting operation

        Examples
        --------
        .. code-block:: python

            import pandas as pd
            from alphapepttools.io.anndata_factory import AnnDataFactory

            # Create sample data
            df = pd.DataFrame(
                {
                    "raw_name": ["sample1", "sample1", "sample2", "sample2"],
                    "protein_group": ["PROT1", "PROT2", "PROT1", "PROT2"],
                    "intensity": [100.5, 200.3, 150.2, 180.7],
                }
            )

            # Initialize factory
            factory = AnnDataFactory(
                psm_df=df, intensity_column="intensity", sample_id_column="raw_name", feature_id_column="protein_group"
            )

            # Create AnnData object and display
            display(factory.create_anndata().to_df())
            display(factory.create_anndata().obs)
            display(factory.create_anndata().var)

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
        adata
            AnnData object to add metadata to
        columns
            Columns to extract as metadata
        index_column
            Column to use as index
        axis
            0 for obs (samples), 1 for var (features)

        Returns
        -------
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
        var_columns
            Additional columns to include in `var` of the AnnData object, by default None
        obs_columns
            Additional columns to include in `obs` of the AnnData object, by default None

        Returns
        -------
        AnnData object where:
        - obs (rows) are samples
        - var (columns) are features (e.g., proteins, peptides, or genes)
        - X contains intensity values

        Examples
        --------
        .. code-block:: python

            import pandas as pd
            from alphapepttools.io.anndata_factory import AnnDataFactory

            # Create sample data with metadata
            df = pd.DataFrame(
                {
                    "raw_name": ["sample1"] * 3 + ["sample2"] * 3,
                    "protein_group": ["PROT1", "PROT2", "PROT3"] * 2,
                    "intensity": [100, 200, 150, 120, 210, 160],
                    "gene_names": ["GENE1", "GENE2", "GENE3"] * 2,
                    "condition": ["control"] * 3 + ["treated"] * 3,
                }
            )

            factory = AnnDataFactory(
                psm_df=df, intensity_column="intensity", sample_id_column="raw_name", feature_id_column="protein_group"
            )

            # Create AnnData with metadata
            adata = factory.create_anndata(
                var_columns=["gene_names"],  # Add gene names to var
                obs_columns=["condition"],  # Add condition to obs
            )

            print(adata.shape)  # (2, 3) - 2 samples, 3 proteins
            print(adata.var["gene_names"])  # Gene annotations
            print(adata.obs["condition"])  # Sample conditions

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

    @classmethod
    def _get_reader_configuration(cls, reader_type: str) -> dict[str, dict[str, Any]]:
        """Get reader-specific configuration for mapping PSMs to anndata."""
        reader_configs = {
            "diann": {
                # Stringent filtering.
                # Defaults to filters by Q value, Library Q value, Global Q value if MBR is active
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
        file_paths
            Path(s) to PSM file(s)
        reader_type
            Type of PSM reader to use, by default "maxquant"
        level
            Level of quantification to read. One of "proteins", "precursors", or "genes". Defaults to "proteins".
        intensity_column
            Name of the column storing intensity data. Default is taken from `psm_reader.yaml`
        feature_id_column
            Name of the column storing feature ids. Default is taken from `psm_reader.yaml`
        sample_id_column
            Name of the column storing sample ids. Default is taken from `psm_reader.yaml`
        additional_columns
            Names of additional columns from the PSM table to retain for experiment-specific metadata.
            These columns can be added to the resulting AnnData object as annotations.
            Note that if a column has a higher cardinality than the `feature_id_column`
            (i.e., multiple values per feature), only the first value encountered will be kept.
        **reader_kwargs
            Additional arguments passed to PSM reader

        Returns
        -------
        Initialized AnnDataFactory instance

        Examples
        --------
        .. code-block:: python

            from alphapepttools.io.anndata_factory import AnnDataFactory

            # Load DIA-NN data at protein level
            # assuming a diann report called "report.tsv" exists in the current directory

            factory = AnnDataFactory.from_files("report.tsv", reader_type="diann", level="proteins")
            adata = factory.create_anndata()

            # Load with custom column names and additional metadata columns
            factory = AnnDataFactory.from_files(
                report_path,
                reader_type="diann",
                intensity_column="Precursor.Quantity",
                additional_columns=["Precursor.Quantity"],  # additional columns need to be specified here.
            )
            adata = factory.create_anndata(
                var_columns=["charge", "sequence"]
            )  # Add m/z and stripped sequence via their alphabase-standardized column names in var
            display(adata.var)  # Check that additional columns are included in var

        """
        # Set default configurations if they are not passed to the function
        reader_config = cls._get_reader_configuration(reader_type)
        for key, value in reader_config.items():
            if key not in reader_kwargs:
                reader_kwargs[key] = value

        reader: PSMReaderBase = psm_reader_provider.get_reader(reader_type, **reader_kwargs)

        # Allow users to add custom columns
        if additional_columns is not None:
            reader.add_column_mapping({col: col for col in additional_columns})
        psm_df = reader.load(file_paths)

        # Get defaults for this reader/level, user input overrides
        defaults = FEATURE_LEVEL_CONFIG.get(level, {})
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
