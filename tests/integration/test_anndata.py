"""Integration tests for anndata creation."""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from alphabase.psm_reader.keys import LibPsmDfCols, PsmDfCols
from alphabase.tools.data_downloader import DataShareDownloader

from alphatools.io.anndata_factory import AnnDataFactory

current_file_directory = Path(__file__).resolve().parent
test_data_path = Path(f"{current_file_directory}/reference_data")


def _assert_reference_df_equal(
    psm_df: pd.DataFrame,
    test_case_name: str,
    *,
    loose_check: bool = False,
    check_psf_df_columns=True,
) -> None:
    """Compare the output of a PSM reader against reference data.

    If reference is not present, save the output as reference data and raise.
    """
    out_file_path = test_data_path / f"reference_{test_case_name}.parquet"
    # psm_df.to_csv(test_data_path / f"reference_{test_case_name}.csv")

    # check that all columns are available in PsmDfCols
    if check_psf_df_columns:
        assert set(psm_df.columns) - set(PsmDfCols.get_values()) - set(LibPsmDfCols.get_values()) == set()

    if out_file_path.exists():
        expected_df = pd.read_parquet(out_file_path)

        # TODO: find out why some results differ in order on the github runner
        if loose_check:
            # check that the data is the same, but ignore the order
            columns_to_sort_by = ["rt", "raw_name"]
            psm_df = psm_df.sort_values(by=columns_to_sort_by).reset_index(drop=True)
            expected_df = expected_df.sort_values(by=columns_to_sort_by).reset_index(drop=True)

        try:
            pd.testing.assert_frame_equal(expected_df, psm_df, check_like=loose_check)
        except AssertionError as e:
            # for whatever reason, columns are int32 on windows runners
            logging.warning(f"Converting int32 to int64 for comparison: {e}")

            for column in psm_df.columns:
                if psm_df[column].dtype == np.int32:
                    psm_df[column] = psm_df[column].astype(np.int64)

            pd.testing.assert_frame_equal(expected_df, psm_df, check_like=loose_check)

    else:
        psm_df.to_parquet(out_file_path)
        # convenience for local development: on the second run, the reference data is available
        raise ValueError("No reference data found.")


def test_anndata_alphadia_181():
    """Test creating anndata from alphadia files."""
    # Create directly from files
    url = "https://datashare.biochem.mpg.de/public.php/dav/files/Hk41INtwBvBl0kP/alphadia_1.8.1_report_head.tsv"
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = DataShareDownloader(url=url, output_dir=temp_dir).download()

        factory = AnnDataFactory.from_files(file_paths=file_path, reader_type="alphadia")

    adata = factory.create_anndata()

    # TODO: compare the whole anndata object here not only the df
    _assert_reference_df_equal(adata.to_df(), "ad_alphadia_181", check_psf_df_columns=False)


def test_anndata_diann_181():
    """Test creating anndata from diann files."""
    # Create directly from files
    url = "https://datashare.biochem.mpg.de/public.php/dav/files/Hk41INtwBvBl0kP/diann_1.8.1_report_head.tsv"
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = DataShareDownloader(url=url, output_dir=temp_dir).download()

        factory = AnnDataFactory.from_files(
            file_paths=file_path,
            reader_type="diann",
            # sample_id_column="File.Name",
            # feature_id_column="Protein.Group",
            # intensity_column="PG.MaxLFQ",
        )

    adata = factory.create_anndata()

    # TODO: compare the whole anndata object here not only the df
    _assert_reference_df_equal(adata.to_df(), "ad_diann_181", check_psf_df_columns=False)


def test_anndata_diann_190():
    """Test creating anndata from diann files."""
    # Create directly from files
    url = "https://datashare.biochem.mpg.de/public.php/dav/files/Hk41INtwBvBl0kP/diann_1.9.0_report_head.tsv"
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = DataShareDownloader(url=url, output_dir=temp_dir).download()

        factory = AnnDataFactory.from_files(
            file_paths=file_path,
            reader_type="diann",
            sample_id_column="File.Name",
            feature_id_column="Protein.Group",
            intensity_column="PG.MaxLFQ",
        )

    adata = factory.create_anndata()

    # TODO: compare the whole anndata object here not only the df
    _assert_reference_df_equal(adata.to_df(), "ad_diann_190", check_psf_df_columns=False)
