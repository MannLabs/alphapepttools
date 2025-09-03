"""Unit tests for alphatools.io.pg_reader module."""

from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.io.pg_reader import SAMPLE_ID_NAME, read_pg_table


class TestReadPGTable:
    """Test suite for read_pg_table function."""

    @pytest.fixture
    def mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        data = {"Sample1": [1.0, 2.0, 3.0, 4.0], "Sample2": [4.0, 5.0, 6.0, np.nan], "Sample3": [7.0, 8.0, 9.0, 10.0]}
        index = pd.Index(["Protein1", "Protein2", "Protein3", "Protein4"], name="protein_id")
        return pd.DataFrame(data, index=index)

    @pytest.fixture
    def mock_reader(self, mock_dataframe):
        """Create a mock reader object."""
        reader = Mock()
        reader.import_file.return_value = mock_dataframe
        return reader

    @pytest.fixture
    def mock_pg_reader_provider(self, mock_reader):
        """Mock the pg_reader_provider."""
        with patch("alphatools.io.pg_reader.pg_reader_provider") as mock_provider:
            mock_provider.get_reader.return_value = mock_reader
            yield mock_provider

    @pytest.mark.parametrize("search_engine", ["alphadia", "alphapept", "diann", "maxquant", "spectronaut"])
    def test_read_pg_table_different_engines(self, mock_pg_reader_provider, search_engine):
        """Test reading PG tables from different search engines."""

        result = read_pg_table("/path/to/file", search_engine)

        # Verify provider was called with correct engine
        mock_pg_reader_provider.get_reader.assert_called_once_with(search_engine)

        # Check result is AnnData object
        assert isinstance(result, ad.AnnData)

    def test_read_pg_table_basic(self, mock_pg_reader_provider, mock_dataframe):
        """Test basic functionality of read_pg_table."""
        result = read_pg_table("/path/to/file", "alphadia")

        # Check dimensions (transposed from features x observations to observations x features)
        assert result.X.shape == (3, 4)  # 3 samples x 3 proteins

        # Check data values (transposed)
        expected_X = mock_dataframe.to_numpy().T
        np.testing.assert_array_equal(result.X, expected_X)

        # Check obs (observations/samples)
        assert SAMPLE_ID_NAME in result.obs.columns
        assert list(result.obs[SAMPLE_ID_NAME]) == ["Sample1", "Sample2", "Sample3"]

    def test_read_pg_table_with_column_mapping(self, mock_pg_reader_provider):
        """Test read_pg_table with column mapping."""
        column_mapping = {"new_col": "old_col", "another_col": "orig_col"}

        result = read_pg_table("/path/to/file", "maxquant", column_mapping=column_mapping)

        # Verify column mapping was passed to get_reader
        mock_pg_reader_provider.get_reader.assert_called_once_with("maxquant", column_mapping=column_mapping)

        assert isinstance(result, ad.AnnData)

    @pytest.mark.parametrize(
        ("measurement_regex", "expected_regex"),
        [("lfq", "lfq"), ("^.*(?<!_LFQ)$", "^.*(?<!_LFQ)$"), ("_intensity$", "_intensity$")],
    )
    def test_read_pg_table_with_measurement_regex(self, mock_pg_reader_provider, measurement_regex, expected_regex):
        """Test read_pg_table with different measurement regex patterns."""
        result = read_pg_table("/path/to/file", "diann", measurement_regex=measurement_regex)

        # Verify measurement_regex was passed correctly
        mock_pg_reader_provider.get_reader.assert_called_once_with("diann", measurement_regex=expected_regex)

        assert isinstance(result, ad.AnnData)

    def test_read_pg_table_full_parameters(self, mock_pg_reader_provider):
        """Test read_pg_table with all parameters specified."""
        column_mapping = {"protein_name": "Protein.Names"}
        measurement_regex = "lfq"

        result = read_pg_table(
            "/path/to/report.tsv", "spectronaut", column_mapping=column_mapping, measurement_regex=measurement_regex
        )

        # Verify all parameters were passed correctly
        mock_pg_reader_provider.get_reader.assert_called_once_with(
            "spectronaut", column_mapping=column_mapping, measurement_regex=measurement_regex
        )

        assert isinstance(result, ad.AnnData)

    def test_read_pg_table_empty_dataframe(self, mock_pg_reader_provider):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        mock_reader = Mock()
        mock_reader.import_file.return_value = empty_df
        mock_pg_reader_provider.get_reader.return_value = mock_reader

        result = read_pg_table("/path/to/empty.csv", "alphapept")

        # Check that empty DataFrame is handled
        assert isinstance(result, ad.AnnData)
        assert result.X.shape[0] == 0 or result.X.shape[1] == 0

    def test_read_pg_table_single_sample(self, mock_pg_reader_provider):
        """Test with single sample DataFrame."""
        single_sample_df = pd.DataFrame(
            {"Sample1": [1.0, 2.0]}, index=pd.Index(["Protein1", "Protein2"], name="protein_id")
        )
        mock_reader = Mock()
        mock_reader.import_file.return_value = single_sample_df
        mock_pg_reader_provider.get_reader.return_value = mock_reader

        result = read_pg_table("/path/to/single.csv", "maxquant")

        assert result.X.shape == (1, 2)  # 1 sample x 2 proteins
        assert len(result.obs) == 1
        assert result.obs[SAMPLE_ID_NAME].iloc[0] == "Sample1"

    def test_var_dataframe_structure(self, mock_pg_reader_provider, mock_dataframe):
        """Test the structure of var DataFrame in AnnData."""
        result = read_pg_table("/path/to/file", "spectronaut")

        # Check var is a DataFrame
        assert isinstance(result.var, pd.DataFrame)

        # Check index name is preserved
        assert result.var.index.name is None  # index=False in to_frame

        # Check protein_id column exists in var
        assert "protein_id" in result.var.columns

    def test_obs_dataframe_structure(self, mock_pg_reader_provider, mock_dataframe):
        """Test the structure of obs DataFrame in AnnData."""
        result = read_pg_table("/path/to/file", "alphapept")

        # Check obs is a DataFrame
        assert isinstance(result.obs, pd.DataFrame)

        # Check sample_id column exists
        assert SAMPLE_ID_NAME in result.obs.columns

        # Check index is default (0, 1, 2, ...)
        assert list(result.obs.index) == list(map(str, (range(len(result.obs)))))

    def test_numeric_data_preservation(self, mock_pg_reader_provider):
        """Test that numeric data is correctly preserved through transformation."""
        # Create DataFrame with specific numeric values
        test_data = pd.DataFrame(
            {
                "S1": [1.234, 5.678, 9.012],
                "S2": [2.345, 6.789, 0.123],
            },
            index=pd.Index(["P1", "P2", "P3"], name="proteins"),
        )

        mock_reader = Mock()
        mock_reader.import_file.return_value = test_data
        mock_pg_reader_provider.get_reader.return_value = mock_reader

        result = read_pg_table("/path/to/data.csv", "diann")

        # Check that values are preserved after transposition
        expected = test_data.to_numpy().T
        np.testing.assert_array_almost_equal(result.X, expected, decimal=6)
