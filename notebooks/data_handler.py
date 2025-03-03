"""An interface to test data, both synthetic and real (downloaded from datashare)."""

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from alphabase.anndata.anndata_factory import AnnDataFactory

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*'progressbar' not installed.*")
    from alphabase.tools.data_downloader import DataShareDownloader

# update this in case the file is moved
REPO_ROOT = Path(__file__).resolve().parent.parent


def create_synthetic_data_3x2() -> ad.AnnData:
    """Create synthetic data with 3 features and 2 cell types."""
    # TODO: improvements: allow to draw from a lognormal distribution for more realistic properties,
    #  dropout simulation based on intensity values, create an arbitrary number of
    #  observations + features

    np.random.seed(0)
    X = np.random.normal(loc=[0, 4, 8], scale=1, size=(100, 3))
    X[np.random.choice(100, 50, replace=False), 0] = np.nan

    # create anndata object
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"celltype": ["A" for _ in range(50)] + ["B" for _ in range(50)]}, index=[f"cell_{i}" for i in range(100)]
        ),
        var=pd.DataFrame({"alternative_name": [f"G{i}" for i in range(3)]}, index=[f"gene_{i}" for i in range(3)]),
    )


class Keys:
    """Keys for accessing the data in TEST_CASES."""

    ADATA = "adata"
    METADATA = "metadata"
    FUNCTION = "function"

    URL = "url"
    LIMIT_KB = "limit_kb"
    FORMAT = "format"


# defines all the test cases
TEST_CASES = {
    "domap": {
        Keys.ADATA: {
            Keys.URL: "https://datashare.biochem.mpg.de/s/sSYkOj22kM5AJ4O/download?path=%2FSearch%20results%2FAlphaDIA%201.9.2%2Fsecond_pass_Lumosmodel&files=precursors.tsv",
            Keys.LIMIT_KB: 1048,
            Keys.FORMAT: "alphadia",  # needs to be a valid reader_type in AnnDataFactory
        },
        Keys.METADATA: {
            Keys.URL: "https://datashare.biochem.mpg.de/s/sSYkOj22kM5AJ4O/download?path=%2F&files=simple_metadata.csv",
            Keys.FORMAT: "csv",
        },
    },
    "synthetic_3x2": {Keys.FUNCTION: create_synthetic_data_3x2},
}


class DataHandler:
    """A class to provide test data, both synthetic and real (downloaded from datashare)."""

    def __init__(self, name: str | None = None, *, target_folder: str | None = None):
        """Initialize with a dictionary of URLs and a target folder.

        Parameters
        ----------
        name
            name of the test case, must be a valid key of TEST_CASES
        target_folder
            Path to the target folder where results will be saved.
            If None, a folder will be created in the root of the repository.
            If "HOME", then a folder will be created in the user's home directory.
        """
        self._test_case = TEST_CASES[name]

        if target_folder is None:
            _target_folder = str(REPO_ROOT / "test_data")
        elif target_folder == "HOME":
            _target_folder = str((Path("~") / "alphatools_test_data").expanduser())
        else:
            _target_folder = target_folder

        self._target_folder = _target_folder

        if self._test_case is None:
            raise ValueError(f"Test case type {name} is not supported, valid options are {TEST_CASES.keys()}.")

    def get_data(self, data_type: str | None = None, *, truncate: bool = False, **kwargs) -> ad.AnnData | str:
        """
        Process a single URL using the existing home-made class.

        Args:
            data_type (str): Type of the data to retrieve.
            truncate (bool): If True, the data will be truncated to the limit specified in TEST_CASES.

        Returns
        -------
            either a string with the downloaded data file or an AnnData object.
        """
        if (function := self._test_case.get(Keys.FUNCTION)) is not None:
            print(f"Creating synthetic data using {function}() .. ")
            return function(**kwargs)

        file_info = self._test_case.get(data_type)

        if file_info is None:
            raise ValueError(f"Data type {data_type} is not supported, valid options are {self._test_case.keys()}.")

        if self._target_folder is None:
            raise ValueError("Target folder is required for downloading data.")

        limit_kb = file_info.get(Keys.LIMIT_KB) if truncate else None
        file_path = DataShareDownloader(file_info[Keys.URL], self._target_folder).download(limit_kb)

        if (file_format := file_info.get(Keys.FORMAT)) == "csv":
            print(f"Creating dataframe from downloaded {file_format} data .. ")
            return pd.read_csv(file_path)

        if file_format is not None:
            print(f"Creating anndata object from downloaded {file_format} data .. ")
            return AnnDataFactory.from_files(file_paths=file_path, reader_type=file_format).create_anndata()

        print("Returning file path to downloaded data .. ")
        return file_path
