"""Unit tests for alphapepttools.io.pg_reader module."""

from unittest.mock import Mock, patch

import pandas as pd

from alphapepttools.io.pg_reader import read_pg_table


class TestReadPGTable:
    """Test suite for read_pg_table function."""

    @patch("alphapepttools.io.pg_reader.pg_reader_provider")
    def test_read_pg_table_default(self, mock_reader_provider):
        """Test that `read_pg_table` properly delegates to alphabase readers with correct default arguments."""
        mock_reader = Mock()
        mock_reader.import_file.return_value = pd.DataFrame()
        mock_reader_provider.get_reader.return_value = mock_reader

        # Test basic usage
        _ = read_pg_table("/path/to/file.tsv", "alphadia")

        mock_reader_provider.get_reader.assert_called_once_with("alphadia")
        mock_reader.import_file.assert_called_once_with("/path/to/file.tsv")

    @patch("alphapepttools.io.pg_reader.pg_reader_provider")
    def test_read_pg_table_custom_arguments(self, mock_reader_provider):
        """Test that `read_pg_table` properly delegates to alphabase readers with correct custom arguments."""
        mock_reader = Mock()
        mock_reader.import_file.return_value = pd.DataFrame()
        mock_reader_provider.get_reader.return_value = mock_reader

        # Test basic usage
        _ = read_pg_table(
            path="/path/to/file.tsv", search_engine="alphadia", column_mapping={"a": "b"}, measurement_regex="test"
        )

        mock_reader_provider.get_reader.assert_called_once_with(
            "alphadia", column_mapping={"a": "b"}, measurement_regex="test"
        )
        mock_reader.import_file.assert_called_once_with("/path/to/file.tsv")
