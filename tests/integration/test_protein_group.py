from pathlib import Path

from alphabase.tools.data_downloader import DataShareDownloader

from alphatools.io import read_pg_matrix

from .test_anndata import _assert_reference_df_equal

current_file_directory = Path(__file__).resolve().parent
test_data_path = Path(f"{current_file_directory}/reference_data")


def test_pg_alphadia_110(tmpdir):
    """Test creating anndata from alphadia protein group table using alphaDIA output."""
    # Create directly from files
    url = "https://datashare.biochem.mpg.de/s/gXVFPZBBkEN5FXi"

    file_path = DataShareDownloader(url=url, output_dir=tmpdir).download()

    adata = read_pg_matrix(file_path=file_path, sample_name="sample_id")

    _assert_reference_df_equal(adata.to_df(), "pg_alphadia110", check_psf_df_columns=False)


def test_pg_rosenberger2023(tmpdir):
    """Test creating anndata from alphadia protein group table using single-cell data by
    Rosenberger et al, 2023."""
    # Create directly from files
    url = "https://datashare.biochem.mpg.de/s/eX2NJ39o9ZaFGMo"

    file_path = DataShareDownloader(url=url, output_dir=tmpdir).download()

    adata = read_pg_matrix(file_path=file_path, sample_name="sample_id")

    _assert_reference_df_equal(adata.to_df(), "pg_rosenberger2023", check_psf_df_columns=False)


def test_pg_diann(tmpdir):
    """Test creating anndata from DIANN protein group table using data by
    Shani Ben-Moshe, 2025."""
    # Create directly from files
    url = "https://datashare.biochem.mpg.de/s/m8q3Lg2ppAWKCft"

    file_path = DataShareDownloader(url=url, output_dir=tmpdir).download()

    adata = read_pg_matrix(
        file_path=file_path,
        sample_name="sample_id",
        feature_metadata_index=[
            0,
            1,
            2,
            3,
        ],
        feature_name=["pg", "protein_names", "genes", "first_protein_description"],
    )

    _assert_reference_df_equal(adata.to_df(), "pg_diann", check_psf_df_columns=False)
