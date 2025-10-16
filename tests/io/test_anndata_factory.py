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
                PsmDfCols.RAW_NAME: ["raw1", "raw1", "raw2"],
                PsmDfCols.PROTEINS: ["protein1", "protein2", "protein1"],
                PsmDfCols.INTENSITY: [100, 200, 300],
                "filesize": [10, 10, 20],
                "filetype": ["raw", "raw", "raw"],
                "gene": ["gene1", "gene2", "gene1"],
                "sequence": ["AAA", "BBB", "AAA"],
            }
        )

    return make_dummy_data()


@pytest.fixture
def test_anndata():
    def make_dummy_data():
        """Return a test AnnData object."""
        return ad.AnnData(
            X=np.array([[100, 200], [300, np.nan]]),
            obs=pd.DataFrame(index=["raw1", "raw2"]),
            var=pd.DataFrame(index=["protein1", "protein2"]),
        )

    return make_dummy_data()


def test_initialization_with_missing_columns(test_psm_df):
    """Test that an error is raised when the input DataFrame is missing required columns."""
    psm_df = test_psm_df.drop(columns=[PsmDfCols.INTENSITY])

    with pytest.raises(ValueError, match=r"Missing required columns: \['intensity'\]"):
        # when
        AnnDataFactory(psm_df)


@pytest.mark.parametrize(
    ("var_columns", "obs_columns", "extra_obs_cols", "extra_var_cols"),
    [
        # 1. No additional columns should work
        (None, None, None, None),
        # 2. Single columns should be correctly added
        ("gene", None, None, pd.DataFrame({"gene": ["gene1", "gene2"]}, index=["protein1", "protein2"])),
        (None, "filesize", pd.DataFrame({"filesize": [10, 20]}, index=["raw1", "raw2"]), None),
        # 3. Multiple columns should be correctly added
        (
            ["gene", "sequence"],
            None,
            None,
            pd.DataFrame({"gene": ["gene1", "gene2"], "sequence": ["AAA", "BBB"]}, index=["protein1", "protein2"]),
        ),
        (
            None,
            ["filesize", "filetype"],
            pd.DataFrame({"filesize": [10, 20], "filetype": ["raw", "raw"]}, index=["raw1", "raw2"]),
            None,
        ),
        # 4. List and string arguments for columns should be handled correctly
        (["gene"], None, None, pd.DataFrame({"gene": ["gene1", "gene2"]}, index=["protein1", "protein2"])),
        (None, ["filesize"], pd.DataFrame({"filesize": [10, 20]}, index=["raw1", "raw2"]), None),
        # 5. Default index match column should be ignored
        (
            ["gene"],
            ["filesize", PsmDfCols.RAW_NAME],
            pd.DataFrame({"filesize": [10, 20]}, index=["raw1", "raw2"]),
            pd.DataFrame({"gene": ["gene1", "gene2"]}, index=["protein1", "protein2"]),
        ),
        (
            ["gene", PsmDfCols.PROTEINS],
            ["filesize"],
            pd.DataFrame({"filesize": [10, 20]}, index=["raw1", "raw2"]),
            pd.DataFrame({"gene": ["gene1", "gene2"]}, index=["protein1", "protein2"]),
        ),
    ],
)
def test_create_anndata_with_valid_dataframe(
    test_psm_df, test_anndata, var_columns, obs_columns, extra_obs_cols, extra_var_cols
):
    """Test that an AnnData object is created correctly from a valid input DataFrame."""
    psm_df = test_psm_df
    factory = AnnDataFactory(psm_df)

    # when
    adata = factory.create_anndata(
        var_columns=var_columns,
        obs_columns=obs_columns,
    )

    # comparison anndata with respective columns added
    comparison_adata = test_anndata.copy()
    if extra_obs_cols is not None:
        comparison_adata = add_metadata(comparison_adata, extra_obs_cols, axis=0)
    if extra_var_cols is not None:
        comparison_adata = add_metadata(comparison_adata, extra_var_cols, axis=1)

    assert adata.shape == (2, 2)
    assert adata.obs.equals(comparison_adata.obs)
    assert adata.var.equals(comparison_adata.var)
    assert np.array_equal(adata.X, np.array([[100, 200], [300, np.nan]]), equal_nan=True)


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
def test_from_files(mock_get_reader_configuration, mock_reader, test_psm_df):
    mock_reader.return_value.load.return_value = test_psm_df

    mock_get_reader_configuration.return_value = {"extra_key": "extra_value"}
    factory = AnnDataFactory.from_files(["file1", "file2"], reader_type="some_reader_type")

    # when
    adata = factory.create_anndata()

    assert adata.shape == (2, 2)
    assert adata.obs_names.tolist() == ["raw1", "raw2"]
    assert adata.var_names.tolist() == ["protein1", "protein2"]
    assert np.array_equal(adata.X, np.array([[100, 200], [300, np.nan]]), equal_nan=True)

    mock_reader.assert_called_once_with("some_reader_type", extra_key="extra_value")


@patch("alphabase.psm_reader.psm_reader.psm_reader_provider.get_reader")
def test_from_files_nan(mock_reader, test_psm_df):
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    PsmDfCols.RAW_NAME: ["raw2"],
                    PsmDfCols.PROTEINS: ["protein2"],
                    PsmDfCols.INTENSITY: [np.nan],
                }
            ),
            test_psm_df,
        ]
    )
    mock_reader.return_value.load.return_value = df

    factory = AnnDataFactory.from_files(["file1", "file2"], reader_type="some_reader_type")

    # when
    adata = factory.create_anndata()

    assert adata.shape == (2, 2)
    assert adata.obs_names.tolist() == ["raw1", "raw2"]
    assert adata.var_names.tolist() == ["protein1", "protein2"]
    assert np.array_equal(adata.X, np.array([[100, 200], [300, np.nan]]), equal_nan=True)

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
