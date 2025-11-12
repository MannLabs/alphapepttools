from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphatools.io import read_pg_table

current_file_directory = Path(__file__).resolve().parent
test_data_path = Path(f"{current_file_directory}/reference_data")


@pytest.fixture
def example_alphadia_tsv(tmp_path) -> tuple[Path, Path]:
    """Get and parse real alphadia PG report matrix."""
    INPUT_DATA_NAME = "alphadia_1.10.4___pg_matrix.tsv"
    REF_DATA_NAME = "reference_ad_alphadia_1.10.4__pg.parquet"

    file_path = test_data_path / INPUT_DATA_NAME
    reference_file_path = test_data_path / REF_DATA_NAME

    return file_path, reference_file_path


@pytest.fixture
def example_alphapept_csv(tmp_path) -> tuple[Path, Path, Path]:
    """Get and parse real alphadia PG report matrix."""

    INPUT_DATA_NAME = "alphapept_0.5.0__pg_matrix.csv"
    REF_DATA_NAME = "reference_ad_alphapept_0.5.0__pg.parquet"
    VAR_DATA_NAME = "reference_ad_var_alphapept_0.5.0__pg.parquet"

    file_path = test_data_path / INPUT_DATA_NAME
    reference_file_path = test_data_path / REF_DATA_NAME
    var_data_path = test_data_path / VAR_DATA_NAME

    return file_path, reference_file_path, var_data_path


def test_alphadia_pg_reader(example_alphadia_tsv: tuple[Path, Path]):
    """Test alphadia reader (basic reader)"""
    file_path, reference_file_path = example_alphadia_tsv

    adata = read_pg_table(file_path, search_engine="alphadia")
    reference_df = pd.read_parquet(reference_file_path)

    pd.testing.assert_frame_equal(adata.to_df(), reference_df)


def test_alphapept_pg_reader(example_alphapept_csv: tuple[Path, Path]):
    """Test alphadia reader (basic reader)"""
    file_path, reference_file_path, var_data_path = example_alphapept_csv

    # Validate that measurement_regex works as expected
    adata = read_pg_table(file_path, search_engine="alphapept", measurement_regex="lfq")

    reference_df = pd.read_parquet(reference_file_path).replace({"": np.nan}).astype(float)
    # Reference .var attribute, validate parsing of metadata
    reference_var = pd.read_parquet(var_data_path)

    pd.testing.assert_frame_equal(adata.to_df(), reference_df)
    pd.testing.assert_frame_equal(adata.var, reference_var)
