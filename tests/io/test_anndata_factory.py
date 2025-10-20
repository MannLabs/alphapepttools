"""Unit tests for the AnnDataFactory class."""

from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from alphabase.psm_reader.keys import PsmDfCols

from alphatools.io.anndata_factory import AnnDataFactory
from alphatools.pp.data import add_metadata


@pytest.fixture
def test_psm_df():
    def make_dummy_data():
        """Return a test PSM DataFrame."""
        return pd.DataFrame(
            {
                # Sample metadata
                "file_id": ["raw1", "raw1", "raw1", "raw2", "raw2", "raw2"],
                "filesize": [10000.0, 10000.0, 10000.0, 20000.0, 20000.0, 20000.0],
                "filetype": ["raw", "raw", "raw", "raw", "raw", "raw"],
                # Feature metadata
                "sequence": ["PEPTIDEK1", "PEPTIDEK2", "PEPTIDEK3", "PEPTIDEK1", "PEPTIDEK2", "PEPTIDEK3"],
                "alternate_gene_name": ["g1", "g1", "g1", "g1", "g1", "g1"],
                # Protein IDs + Intensities
                "protein_id": ["protein1", "protein1", "protein2", "protein1", "protein1", "protein2"],
                "protein_intensity": [np.nan, 100.0, np.nan, 200.0, np.nan, 300.0],
                # Precursor IDs + Intensities
                "precursor_id": ["precursor1", "precursor2", "precursor3", "precursor1", "precursor2", "precursor3"],
                "precursor_intensity": [50.0, 150.0, 200.0, 150.0, 250.0, 200.0],
                # Gene IDs + Intensities
                "gene_id": ["gene1", "gene1", "gene1", "gene1", "gene1", "gene1"],
                "gene_intensity": [1000.0, 1000.0, 1000.0, 2000.0, 2000.0, 2000.0],
                # MaxQuant-like column
                "Retention time": [1, 2, 3, 2, 3, 4],
            }
        )

    return make_dummy_data()


@pytest.fixture
def test_protein_anndata():
    def make_dummy_data():
        """Return the pivoted protein AnnData object."""
        return ad.AnnData(
            X=np.array([[100.0, np.nan], [200.0, 300.0]]),
            obs=pd.DataFrame(index=["raw1", "raw2"]),
            var=pd.DataFrame(index=["protein1", "protein2"]),
        )

    return make_dummy_data()


@pytest.fixture
def test_precursor_anndata():
    def make_dummy_data():
        """Return a test AnnData object"""
        return ad.AnnData(
            X=np.array([[50.0, 150.0, 200.0], [150.0, 250.0, 200.0]]),
            obs=pd.DataFrame(index=["raw1", "raw2"]),
            var=pd.DataFrame(index=["precursor1", "precursor2", "precursor3"]),
        )

    return make_dummy_data()


@pytest.fixture
def test_gene_anndata():
    def make_dummy_data():
        """Return the pivoted gene AnnData object."""
        return ad.AnnData(
            X=np.array([[1000.0], [2000.0]]),
            obs=pd.DataFrame(index=["raw1", "raw2"]),
            var=pd.DataFrame(index=["gene1"]),
        )

    return make_dummy_data()


@pytest.mark.parametrize(
    ("var_columns", "obs_columns"),
    [
        # 1. No additional columns should work
        (None, None),
        # 2. Single columns should be correctly added
        ("sequence", None),
        # 3. Single columns specified as list should be correctly added
        (None, ["filesize"]),
        # 4. Multiple columns should be correctly added
        (["alternate_gene_name", "sequence"], None),
        (None, ["filesize", "filetype"]),
    ],
)
@pytest.mark.parametrize(
    ("feature_id_column", "intensity_column", "sample_id_column"),
    [
        # 1. Standard pivoting case: proteins
        ("protein_id", "protein_intensity", "file_id"),
        # 2. Precursors
        ("precursor_id", "precursor_intensity", "file_id"),
        # 3. Genes
        ("gene_id", "gene_intensity", "file_id"),
    ],
)
def test_create_anndata_with_valid_dataframe(
    test_psm_df,
    feature_id_column,
    intensity_column,
    sample_id_column,
    test_protein_anndata,
    test_gene_anndata,
    test_precursor_anndata,
    var_columns,
    obs_columns,
):
    """Test that an AnnData object is created correctly from a valid input DataFrame."""
    psm_df = test_psm_df

    factory = AnnDataFactory(
        psm_df=psm_df,
        intensity=intensity_column,
        sample_id=sample_id_column,
        feature_id=feature_id_column,
    )

    adata = factory.create_anndata(
        var_columns=var_columns,
        obs_columns=obs_columns,
    )

    # Obtain the correct comparison anndata based on feature_id_column
    if feature_id_column == "protein_id":
        comparison_adata = test_protein_anndata.copy()
    elif feature_id_column == "precursor_id":
        comparison_adata = test_precursor_anndata.copy()
    elif feature_id_column == "gene_id":
        comparison_adata = test_gene_anndata.copy()
    else:
        raise ValueError("Invalid feature_column")

    # Add respective metadata columns
    if obs_columns is not None:
        extra_obs_cols = [obs_columns] if isinstance(obs_columns, str) else obs_columns
        extra_obs_df = psm_df[[sample_id_column, *extra_obs_cols]].set_index(sample_id_column, drop=True)
        extra_obs_df = extra_obs_df[~extra_obs_df.index.duplicated(keep="first")]
        comparison_adata = add_metadata(comparison_adata, extra_obs_df, axis=0)
    if var_columns is not None:
        extra_var_cols = [var_columns] if isinstance(var_columns, str) else var_columns
        extra_var_df = psm_df[[feature_id_column, *extra_var_cols]].set_index(feature_id_column, drop=True)
        extra_var_df = extra_var_df[~extra_var_df.index.duplicated(keep="first")]
        comparison_adata = add_metadata(comparison_adata, extra_var_df, axis=1)

    assert adata.obs.equals(comparison_adata.obs)
    assert adata.var.equals(comparison_adata.var)
    assert adata.to_df().equals(comparison_adata.to_df())


def test_create_anndata_with_missing_intensity_values():
    """Test that missing intensity values are replaced with NaNs in the AnnData object."""
    psm_df = pd.DataFrame(
        {
            PsmDfCols.RAW_NAME: ["raw1", "raw2"],
            PsmDfCols.PROTEINS: ["protein1", "protein2"],
            PsmDfCols.INTENSITY: [100, np.nan],
        }
    )
    factory = AnnDataFactory(psm_df)

    # when
    adata = factory.create_anndata()

    assert adata.shape == (2, 2)
    assert adata.obs_names.tolist() == ["raw1", "raw2"]
    assert adata.var_names.tolist() == ["protein1", "protein2"]
    assert np.array_equal(adata.X, np.array([[100.0, np.nan], [np.nan, np.nan]]), equal_nan=True)


def test_create_anndata_with_duplicate_proteins():
    """Test that intensity values for duplicate proteins in the same raw file are aggregated correctly."""
    psm_df = pd.DataFrame(
        {
            PsmDfCols.RAW_NAME: ["raw1", "raw1", "raw2"],
            PsmDfCols.PROTEINS: ["protein1", "protein1", "protein1"],
            PsmDfCols.INTENSITY: [100, 200, 300],
        }
    )
    factory = AnnDataFactory(psm_df)

    # when
    adata = factory.create_anndata()

    assert adata.shape == (2, 1)
    assert adata.obs_names.tolist() == ["raw1", "raw2"]
    assert adata.var_names.tolist() == ["protein1"]
    assert np.array_equal(
        adata.X,
        np.array([[100], [300]]),  # first is taken
    )


def test_create_anndata_with_empty_dataframe():
    """Test that an empty AnnData object is created when the input DataFrame is empty."""
    psm_df = pd.DataFrame(columns=[PsmDfCols.RAW_NAME, PsmDfCols.PROTEINS, PsmDfCols.INTENSITY])
    factory = AnnDataFactory(psm_df)

    # when
    adata = factory.create_anndata()

    assert adata.shape == (0, 0)


@patch("alphabase.psm_reader.psm_reader.psm_reader_provider.get_reader")
@patch("alphatools.io.anndata_factory.AnnDataFactory._get_reader_configuration")
def test_from_files(mock_get_reader_configuration, mock_reader, test_psm_df, test_protein_anndata):
    mock_reader.return_value.load.return_value = test_psm_df.rename(
        columns={
            "protein_intensity": PsmDfCols.INTENSITY,
            "protein_id": PsmDfCols.PROTEINS,
            "file_id": PsmDfCols.RAW_NAME,
        }
    )

    mock_get_reader_configuration.return_value = {"extra_key": "extra_value"}

    factory = AnnDataFactory.from_files(
        file_paths=["file1", "file2"],
        reader_type="diann",
        intensity_column=PsmDfCols.INTENSITY,
        feature_id_column=PsmDfCols.PROTEINS,
        sample_id_column=PsmDfCols.RAW_NAME,
    )

    # when
    adata = factory.create_anndata()

    comparison_adata = test_protein_anndata
    assert adata.obs.equals(comparison_adata.obs)
    assert adata.var.equals(comparison_adata.var)
    assert adata.to_df().equals(comparison_adata.to_df())

    mock_reader.assert_called_once_with("diann", extra_key="extra_value")


@patch("alphabase.psm_reader.psm_reader.psm_reader_provider.get_reader")
def test_from_files_nan(mock_reader, test_psm_df, test_protein_anndata):
    mock_reader.return_value.load.return_value = test_psm_df.rename(
        columns={
            "protein_intensity": PsmDfCols.INTENSITY,
            "protein_id": PsmDfCols.PROTEINS,
            "file_id": PsmDfCols.RAW_NAME,
        }
    )

    factory = AnnDataFactory.from_files(["file1", "file2"], reader_type="some_reader_type")

    # when
    adata = factory.create_anndata()

    comparison_adata = test_protein_anndata
    assert adata.obs.equals(comparison_adata.obs)
    assert adata.var.equals(comparison_adata.var)
    assert adata.to_df().equals(comparison_adata.to_df())

    mock_reader.assert_called_once_with("some_reader_type")


def test_get_reader_configuration_with_valid_reader_type():
    """Test that the correct configuration is returned for a valid reader type."""
    # when
    config = AnnDataFactory._get_reader_configuration("diann")  # diann is taken as an example here

    assert config == {
        "filter_first_search_fdr": True,
        "filter_second_search_fdr": True,
    }


def test_get_reader_configuration_with_unknown_reader_type():
    """Test that a reader type without special config is handled correctly."""
    # when
    config = AnnDataFactory._get_reader_configuration("invalid_reader_type")
    assert config == {}
