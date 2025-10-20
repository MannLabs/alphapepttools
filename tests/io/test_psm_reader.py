from unittest.mock import Mock, patch

from alphatools.io import read_psm_table


class TestReadPsmTable:
    """Test suite for read_psm_table function."""

    @patch("alphatools.io.psm_reader.AnnDataFactory")
    def test_read_psm_table_calls_factory_with_correct_args(self, mock_factory_class):
        """Test that read_psm_table properly delegates to AnnDataFactory with correct arguments."""
        # Setup mock
        mock_factory_instance = Mock()
        mock_anndata = Mock()
        mock_factory_instance.create_anndata.return_value = mock_anndata
        mock_factory_class.from_files.return_value = mock_factory_instance

        result = read_psm_table("/path/to/file.txt", "alphadia")

        mock_factory_class.from_files.assert_called_once_with(
            file_paths="/path/to/file.txt",
            reader_type="alphadia",
            level="proteins",
            intensity_column=None,
            feature_id_column=None,
            sample_id_column=None,
        )
        mock_factory_instance.create_anndata.assert_called_once()
        assert result == mock_anndata

    @patch("alphatools.io.psm_reader.AnnDataFactory")
    def test_read_psm_table_with_all_parameters(self, mock_factory_class):
        """Test that all parameters are correctly passed through to AnnDataFactory."""
        # Setup mock
        mock_factory_instance = Mock()
        mock_factory_class.from_files.return_value = mock_factory_instance

        # Test with all parameters
        read_psm_table(
            file_paths=["/path/to/file1.txt", "/path/to/file2.txt"],
            search_engine="maxquant",
            level="precursors",
            intensity_column="custom_intensity",
            feature_id_column="custom_feature",
            sample_id_column="custom_sample",
            extra_param="extra_value",
        )

        mock_factory_class.from_files.assert_called_once_with(
            file_paths=["/path/to/file1.txt", "/path/to/file2.txt"],
            reader_type="maxquant",
            level="precursors",
            intensity_column="custom_intensity",
            feature_id_column="custom_feature",
            sample_id_column="custom_sample",
            extra_param="extra_value",
        )
