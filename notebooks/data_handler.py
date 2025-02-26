"""An interface to test data, both synthetic and real (downloaded from datashare)."""

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from alphabase.tools.data_downloader import DataShareDownloader

from alphatools.io.anndata_factory import AnnDataFactory


def create_synthetic_data1() -> ad.AnnData:
    """Create synthetic data with 3 features and 2 cell types."""
    np.random.seed(0)
    feature_1 = np.random.normal(0, 1, 100)
    feature_2 = np.random.normal(0, 1, 100) + 4
    feature_3 = np.random.normal(0, 1, 100) + 8

    # replace 50 % of values with NaN in first feature
    feature_1[np.random.choice(100, 50, replace=False)] = np.nan

    # create anndata object
    return ad.AnnData(
        X=np.vstack((feature_1, feature_2, feature_3)).T,
        obs=pd.DataFrame(
            {"celltype": ["A" for _ in range(50)] + ["B" for _ in range(50)]}, index=[f"cell_{i}" for i in range(100)]
        ),
        var=pd.DataFrame({"alternative_name": [f"G{i}" for i in range(3)]}, index=[f"gene_{i}" for i in range(3)]),
    )


class Keys:
    """Keys for the first layer of the TEST_CASES."""

    ADATA = "adata"
    METADATA = "metadata"
    FUNCTION = "function"


# defines all the test cases
TEST_CASES = {
    "domap": {
        Keys.ADATA: {
            "url": "https://datashare.biochem.mpg.de/s/sSYkOj22kM5AJ4O/download?path=%2FSearch%20results%2FAlphaDIA%201.9.2%2Fsecond_pass_Lumosmodel&files=precursors.tsv",
            "limit_kb": 1048,
            "format": "alphadia",
        },
        Keys.METADATA: {
            "url": "https://datashare.biochem.mpg.de/s/sSYkOj22kM5AJ4O/download?path=%2F&files=simple_metadata.csv",
            "format": "csv",
        },
    },
    "synthetic1": {Keys.FUNCTION: create_synthetic_data1},
}


class DataHandler:
    """A class to provide test data, both synthetic and real (downloaded from datashare)."""

    def __init__(self, name: str | None = None, *, target_folder: str | None = None):
        """
        Initialize with a dictionary of URLs and a target folder.

        The environment parameters `_TEST_NAME` and `_TEST_TARGET_FOLDER` override the
        respective arguments.

        Args:
            name (str): name of the test case
            target_folder (str): Path to the target folder where results will be saved. If "HOME", then a
            folder will be created in the user's home directory. Must be `None` only if synthetic data is used.
        """
        _name_from_env = os.environ.get("_TEST_NAME")
        _test_case_name = _name_from_env if _name_from_env else name
        self._test_case = TEST_CASES[_test_case_name]

        _target_folder_from_env = os.environ.get("_TEST_TARGET_FOLDER")
        if _target_folder_from_env:
            _target_folder = _target_folder_from_env
        elif target_folder == "HOME":
            _target_folder = str((Path("~") / "alphatools_test_data").expanduser())
        else:
            _target_folder = target_folder

        self._target_folder = _target_folder

        if self._test_case is None:
            raise ValueError(
                f"Test case type {_test_case_name} is not supported, valid options are {TEST_CASES.keys()}."
            )

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

        limit_kb = file_info.get("limit_kb") if truncate else None
        file_path = DataShareDownloader(file_info["url"], self._target_folder).download(limit_kb)

        if (file_format := file_info.get("format")) == "csv":
            print(f"Creating dataframe downloaded {file_format} data .. ")
            return pd.read_csv(file_path)

        if file_format is not None:
            print(f"Creating anndata object from downloaded {file_format} data .. ")
            return AnnDataFactory.from_files(file_paths=file_path, reader_type=file_format).create_anndata()

        print("Returning file path to downloaded data .. ")
        return file_path
