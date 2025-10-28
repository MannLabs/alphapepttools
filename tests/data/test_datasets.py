from unittest.mock import Mock, patch

import pandas as pd
import pytest

from alphatools.data.datasets import StudyCollection, StudyData, available_data, get_data, study_collection


@pytest.fixture
def minimal_study():
    """Create a minimal StudyData instance for testing."""
    return StudyData(
        name="test_study",
        url="https://example.com/data",
        search_engine="test_engine",
        data_type="pg",
    )


@pytest.fixture
def full_study():
    """Create a StudyData instance with all optional fields."""
    return StudyData(
        name="full_test_study",
        url="https://example.com/full_data",
        search_engine="test_engine",
        data_type="psm",
        citation="Test Citation 2024",
        description="A test study with full metadata",
    )


@pytest.fixture
def empty_collection():
    """Create an empty StudyCollection instance."""
    return StudyCollection()


@pytest.fixture
def populated_collection(minimal_study, full_study):
    """Create a StudyCollection with two studies."""
    collection = StudyCollection()
    collection.register([minimal_study, full_study])
    return collection


def test_study_data_can_be_created_with_minimal_fields() -> None:
    """Test that StudyData can be instantiated with only required fields."""
    # given
    name = "minimal_test"
    url = "https://test.com/data"
    search_engine = "test_engine"
    data_type = "pg"

    # when
    study = StudyData(
        name=name,
        url=url,
        search_engine=search_engine,
        data_type=data_type,
    )

    # then
    assert study.name == name
    assert study.url == url
    assert study.search_engine == search_engine
    assert study.data_type == data_type
    assert study.citation is None
    assert study.description is None


def test_study_data_can_be_created_with_all_fields() -> None:
    """Test that StudyData can be instantiated with all fields including optional ones."""
    # given
    name = "full_test"
    url = "https://test.com/full_data"
    search_engine = "full_engine"
    data_type = "study_psm"
    citation = "Full Citation 2025"
    description = "Complete description of the study"

    # when
    study = StudyData(
        name=name,
        url=url,
        search_engine=search_engine,
        data_type=data_type,
        citation=citation,
        description=description,
    )

    # then
    assert study.name == name
    assert study.url == url
    assert study.search_engine == search_engine
    assert study.data_type == data_type
    assert study.citation == citation
    assert study.description == description


def test_study_data_download_should_call_datashare_downloader(minimal_study, tmp_path) -> None:
    """Test that StudyData.download calls DataShareDownloader with correct parameters."""
    output_dir = tmp_path / "test_output"
    expected_result = tmp_path / "test_output" / "data.zip"
    mock_downloader = Mock()
    mock_downloader.download.return_value = expected_result

    with patch("alphatools.data.datasets.DataShareDownloader", return_value=mock_downloader) as mock_class:
        result = minimal_study.download(output_dir)

    mock_class.assert_called_once_with(url="https://example.com/data", output_dir=output_dir)
    mock_downloader.download.assert_called_once()
    assert result == expected_result


def test_study_collection_get_study_should_return_study_when_name_exists(populated_collection, minimal_study) -> None:
    """Test that get_study returns the correct study when it exists in the collection."""
    result = populated_collection.get_study("test_study")

    assert result == minimal_study
    assert result.name == "test_study"
    assert result.url == "https://example.com/data"


def test_study_collection_get_study_should_raise_keyerror_when_name_not_found(populated_collection) -> None:
    """Test that get_study raises KeyError when study name doesn't exist."""
    non_existent_name = "non_existent_study"

    with pytest.raises(KeyError, match="Study non_existent_study not found"):
        populated_collection.get_study(non_existent_name)


def test_study_collection_register_single_study(empty_collection, minimal_study) -> None:
    """Test that register can add a single StudyData instance."""
    assert len(empty_collection) == 0

    empty_collection.register(minimal_study)

    assert len(empty_collection) == 1
    assert empty_collection.get_study("test_study") == minimal_study


def test_study_collection_register_should_accept_list_of_studies(empty_collection, minimal_study, full_study) -> None:
    """Test that register can add multiple StudyData instances at once."""
    assert len(empty_collection) == 0
    studies = [minimal_study, full_study]

    empty_collection.register(studies)

    assert len(empty_collection) == len(studies)
    assert empty_collection.get_study("test_study") == minimal_study
    assert empty_collection.get_study("full_test_study") == full_study


def test_available_data_should_return_global_study_collection() -> None:
    """Test that available_data returns the module-level study collection."""
    # given / when
    result = available_data()

    # then
    assert isinstance(result, StudyCollection)
    assert result is study_collection
    assert len(result) > 0


def test_get_data_should_download_from_correct_study(tmp_path) -> None:
    """Test that get_data retrieves and downloads from the correct study."""
    # given
    study_name = "bader2020_pg_alphadia"
    output_dir = tmp_path / "downloads"
    expected_path = tmp_path / "downloads" / "data.tsv"
    mock_downloader = Mock()
    mock_downloader.download.return_value = expected_path

    # when
    with patch("alphatools.data.datasets.DataShareDownloader", return_value=mock_downloader):
        result = get_data(study_name, output_dir)

    # then
    assert result == expected_path
    mock_downloader.download.assert_called_once()


def test_get_data_should_raise_keyerror_when_study_not_found() -> None:
    """Test that get_data raises KeyError when study name doesn't exist."""
    # given
    non_existent_study = "non_existent_study_name"

    # when / then
    with pytest.raises(KeyError, match="Study non_existent_study_name not found"):
        get_data(non_existent_study)


def test_study_collection_contains_all_expected_studies() -> None:
    """Test that module-level collection has all expected predefined studies."""
    # given
    expected_studies = [
        "bader2020_pg_alphadia",
        "bader2020_pg_diann",
        "spectronaut_pg",
        "bader2020_psm_alphadia",
        "bader2020_psm_diann",
        "spectronaut_psm",
    ]

    # when
    collection = available_data()

    for study_name in expected_studies:
        study = collection.get_study(study_name)
        assert study.name == study_name
        assert isinstance(study, StudyData)


def test_study_data_download_should__use_cwd(minimal_study) -> None:
    """Test that StudyData.download uses current working directory when output_dir is None."""
    # given
    mock_downloader = Mock()
    mock_downloader.download.return_value = Mock()

    # when
    with (
        patch("alphatools.data.datasets.DataShareDownloader", return_value=mock_downloader) as mock_class,
        patch("alphatools.data.datasets.Path.cwd") as mock_cwd,
    ):
        mock_cwd.return_value = Mock()
        minimal_study.download(output_dir=None)

    # then
    mock_cwd.assert_called_once()
    mock_class.assert_called_once()
    call_args = mock_class.call_args
    assert call_args.kwargs["output_dir"] == mock_cwd.return_value


def test_study_data_df_property_should_return_dataframe_with_all_fields(full_study) -> None:
    """Test that StudyData.df returns a DataFrame with all study information."""
    # given
    expected_df = pd.DataFrame(
        data=[
            {
                "name": "full_test_study",
                "url": "https://example.com/full_data",
                "search_engine": "test_engine",
                "data_type": "psm",
                "citation": "Test Citation 2024",
                "description": "A test study with full metadata",
            }
        ]
    )

    # when
    result_df = full_study.df

    # then
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_study_data_df_property_should_return_dataframe_with_none_for_optional_fields(minimal_study) -> None:
    """Test that StudyData.df returns a DataFrame with None values for optional fields."""
    # given
    expected_df = pd.DataFrame(
        data=[
            {
                "name": "test_study",
                "url": "https://example.com/data",
                "search_engine": "test_engine",
                "data_type": "pg",
                "citation": None,
                "description": None,
            }
        ]
    )

    # when
    result_df = minimal_study.df

    # then
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_study_collection_df_property_should_concatenate_all_studies(
    populated_collection, minimal_study, full_study
) -> None:
    """Test that StudyCollection.df returns a concatenated DataFrame of all studies."""
    # given
    expected_df = pd.concat([minimal_study.df, full_study.df], ignore_index=True)

    # when
    result_df = populated_collection.df

    # then
    pd.testing.assert_frame_equal(result_df, expected_df)
