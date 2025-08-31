from unittest.mock import Mock, patch

import anndata as ad
import pytest

from alphatools.io import read_psm_table


class TestReadPsmTable:
    """Test suite for read_psm_table function."""

    @pytest.fixture
    def mock_anndata_factory(self):
        """Mock AnnDataFactory and its methods."""
        with patch("alphatools.io.psm_reader.AnnDataFactory") as mock_factory_class:
            mock_factory_instance = Mock()
            mock_anndata = Mock(spec=ad.AnnData)
            mock_factory_instance.create_anndata.return_value = mock_anndata
            mock_factory_class.from_files.return_value = mock_factory_instance
            yield mock_factory_class, mock_factory_instance, mock_anndata

    def test_read_psm_table_single_file_path(self, mock_anndata_factory):
        """Test reading PSM table with single file path."""
        mock_factory_class, mock_factory_instance, mock_anndata = mock_anndata_factory

        file_path = "/path/to/psm_table.txt"
        search_engine = "alphadia"

        result = read_psm_table(file_path, search_engine)

        # Verify AnnDataFactory.from_files was called with correct arguments
        mock_factory_class.from_files.assert_called_once_with(
            file_paths=file_path,
            reader_type=search_engine,
            intensity_column=None,
            protein_id_column=None,
            raw_name_column=None,
        )

        # Verify create_anndata was called
        mock_factory_instance.create_anndata.assert_called_once()

        # Verify correct return value
        assert result == mock_anndata

    def test_read_psm_table_multiple_file_paths(self, mock_anndata_factory):
        """Test reading PSM table with multiple file paths."""
        mock_factory_class, mock_factory_instance, mock_anndata = mock_anndata_factory

        file_paths = ["/path/to/psm1.txt", "/path/to/psm2.txt", "/path/to/psm3.txt"]
        search_engine = "maxquant"

        result = read_psm_table(file_paths, search_engine)

        mock_factory_class.from_files.assert_called_once_with(
            file_paths=file_paths,
            reader_type=search_engine,
            intensity_column=None,
            protein_id_column=None,
            raw_name_column=None,
        )

        mock_factory_instance.create_anndata.assert_called_once()
        assert result == mock_anndata

    def test_read_psm_table_with_custom_columns(self, mock_anndata_factory):
        """Test reading PSM table with custom column specifications."""
        mock_factory_class, mock_factory_instance, mock_anndata = mock_anndata_factory

        file_path = "/path/to/custom_psm.txt"
        search_engine = "diann"
        intensity_col = "custom_intensity"
        feature_id_col = "custom_feature_id"
        sample_id_col = "custom_sample_id"

        result = read_psm_table(
            file_path,
            search_engine,
            intensity_column=intensity_col,
            feature_id_column=feature_id_col,
            sample_id_column=sample_id_col,
        )

        mock_factory_class.from_files.assert_called_once_with(
            file_paths=file_path,
            reader_type=search_engine,
            intensity_column=intensity_col,
            protein_id_column=feature_id_col,
            raw_name_column=sample_id_col,
        )

        mock_factory_instance.create_anndata.assert_called_once()
        assert result == mock_anndata

    def test_read_psm_table_with_kwargs(self, mock_anndata_factory):
        """Test reading PSM table with additional keyword arguments."""
        mock_factory_class, mock_factory_instance, mock_anndata = mock_anndata_factory

        file_path = "/path/to/psm.txt"
        search_engine = "spectronaut"
        extra_kwargs = {
            "some_param": "value1",
            "another_param": 42,
            "bool_param": True,
        }

        result = read_psm_table(file_path, search_engine, **extra_kwargs)

        mock_factory_class.from_files.assert_called_once_with(
            file_paths=file_path,
            reader_type=search_engine,
            intensity_column=None,
            protein_id_column=None,
            raw_name_column=None,
            **extra_kwargs,
        )

        mock_factory_instance.create_anndata.assert_called_once()
        assert result == mock_anndata

    def test_read_psm_table_with_all_parameters(self, mock_anndata_factory):
        """Test reading PSM table with all parameters specified."""
        mock_factory_class, mock_factory_instance, mock_anndata = mock_anndata_factory

        file_paths = ["/path/to/psm1.txt", "/path/to/psm2.txt"]
        search_engine = "alphapept"
        intensity_col = "intensity_col"
        feature_id_col = "feature_col"
        sample_id_col = "sample_col"
        extra_kwargs = {"param1": "value1", "param2": 123}

        result = read_psm_table(
            file_paths,
            search_engine,
            intensity_column=intensity_col,
            feature_id_column=feature_id_col,
            sample_id_column=sample_id_col,
            **extra_kwargs,
        )

        mock_factory_class.from_files.assert_called_once_with(
            file_paths=file_paths,
            reader_type=search_engine,
            intensity_column=intensity_col,
            protein_id_column=feature_id_col,
            raw_name_column=sample_id_col,
            **extra_kwargs,
        )

        mock_factory_instance.create_anndata.assert_called_once()
        assert result == mock_anndata
