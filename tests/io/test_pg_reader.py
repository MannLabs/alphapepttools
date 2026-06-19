"""Unit tests for alphapepttools.io.pg_reader module."""

from unittest.mock import Mock, patch

import pandas as pd

from alphapepttools.io.pg_reader import read_pg_table


class TestReadPGTable:
    """Test suite for read_pg_table function."""

    @patch("alphapepttools.io.pg_reader.pg_reader_provider")
    def test_read_pg_table__default(self, mock_reader_provider):
        """Test that `read_pg_table` properly delegates to alphabase readers with correct default arguments."""
        mock_reader = Mock()
        mock_reader.import_file.return_value = pd.DataFrame()
        mock_reader_provider.get_reader.return_value = mock_reader

        # Test basic usage
        _ = read_pg_table("/path/to/file.tsv", "alphadia")

        mock_reader_provider.get_reader.assert_called_once_with("alphadia")
        mock_reader.import_file.assert_called_once_with("/path/to/file.tsv")

    @patch("alphapepttools.io.pg_reader.pg_reader_provider")
    def test_read_pg_table__custom_arguments(self, mock_reader_provider):
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

    @patch("alphapepttools.io.pg_reader.pg_reader_provider")
    def test_read_pg_table__add_column_mapping(self, mock_reader_provider):
        """Test that `read_pg_table` properly delegates to alphabase readers with correct custom arguments."""
        mock_reader = Mock()
        mock_reader.import_file.return_value = pd.DataFrame()
        mock_reader_provider.get_reader.return_value = mock_reader

        # Test basic usage
        _ = read_pg_table(
            path="/path/to/file.tsv",
            search_engine="alphadia",
            additional_column_mapping={"custom_new_name": "specific_report_column"},
            column_mapping={"a": "b"},
            measurement_regex="test",
        )

        mock_reader_provider.get_reader.assert_called_once_with(
            "alphadia", column_mapping={"a": "b"}, measurement_regex="test"
        )
        mock_reader.add_column_mapping.assert_called_once_with({"custom_new_name": "specific_report_column"})
        mock_reader.import_file.assert_called_once_with("/path/to/file.tsv")

    @patch("alphapepttools.io.pg_reader.pg_reader_provider")
    def test_read_pg_table__multiindex_unique_first_level(self, mock_reader_provider):
        """MultiIndex with unique first level: use it as var index, remaining levels as var columns."""
        mock_reader = Mock()
        mock_reader.import_file.return_value = pd.DataFrame(
            {"sample1": [100.0, 200.0], "sample2": [150.0, 250.0]},
            index=pd.MultiIndex.from_tuples(
                [("protein1", "gene_a"), ("protein2", "gene_b")],
                names=["protein_id", "gene_name"],
            ),
        )
        mock_reader_provider.get_reader.return_value = mock_reader

        adata = read_pg_table("/path/to/file.tsv", "alphadia")

        # Features (df rows) become var; samples (df cols) become obs
        assert adata.var_names.tolist() == ["protein1", "protein2"]
        assert adata.var["gene_name"].tolist() == ["gene_a", "gene_b"]
        assert adata.obs_names.tolist() == ["sample1", "sample2"]

    @patch("alphapepttools.io.pg_reader.pg_reader_provider")
    def test_read_pg_table__non_unique_first_level(self, mock_reader_provider):
        """Non-unique first level: flatten all index levels to var columns with integer var index."""
        mock_reader = Mock()
        mock_reader.import_file.return_value = pd.DataFrame(
            {"sample1": [100.0, 200.0, 300.0], "sample2": [150.0, 250.0, 350.0]},
            index=pd.Index(["protein1", "protein1", "protein2"], name="protein_id"),
        )
        mock_reader_provider.get_reader.return_value = mock_reader

        adata = read_pg_table("/path/to/file.tsv", "alphadia")

        # Non-unique → original index becomes a var column; var index falls back to integers
        assert len(adata.var) == 3  # noqa: PLR2004
        assert adata.var["protein_id"].tolist() == ["protein1", "protein1", "protein2"]
        assert adata.obs_names.tolist() == ["sample1", "sample2"]
