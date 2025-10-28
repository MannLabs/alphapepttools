"""Example datasets"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from alphabase.tools.data_downloader import DataShareDownloader

STUDY_CONFIG = Path(__file__).parent / "datasets.yaml"


@dataclass
class StudyData:
    """Base class that represents

    Parameters
    ----------
    name
        Unique study name
    url
        URL/datashare link where data is stored
    search_engine
        Name of search engine with which the data was searched
    citation
        Citation of study
    data_type
        Type of data
            - pg: Protein group (PG) table
            - psm: Peptide spectrum match (PSM) table
            - study_psm: Full study data (metadata + PSM table)
            - study_pg: Full study data (metadata + PG table)
    """

    name: str
    url: str
    search_engine: str
    data_type: Literal["pg", "psm", "study_psm", "study_pg"]
    citation: str | None = None
    description: str | None = None

    def download(self, output_dir: str | Path | None = None) -> Path:
        """Download data"""
        output_dir = Path.cwd() if output_dir is None else output_dir
        return DataShareDownloader(url=self.url, output_dir=output_dir).download()

    @property
    def df(self) -> pd.DataFrame:
        """Representation of study data as dataframe"""
        return pd.DataFrame(
            data=[
                {
                    "name": self.name,
                    "url": self.url,
                    "search_engine": self.search_engine,
                    "data_type": self.data_type,
                    "citation": self.citation,
                    "description": self.description,
                }
            ]
        )

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks"""
        description = f"<p><strong>Description:</strong> {self.description}</p>" if self.description else ""
        citation = f"<p><strong>Citation:</strong> <small>{self.citation}</small></p>" if self.citation else ""

        html = f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0;">
            <h3>{self.name}</h3>
            <p><strong>Data Type:</strong> {self.data_type}</p>
            <p><strong>Search Engine:</strong> {self.search_engine}</p>
            {description}
            {citation}
        </div>
        """
        return html  # noqa: RET504 Remove unnecessary assignment - return value would be very long


class StudyCollection:
    """Collection of all studies"""

    def __init__(self):
        self.collection: list[StudyData] = []

    def get_study(self, name: str) -> StudyData:
        """Get study by name"""
        for study in self.collection:
            if study.name == name:
                return study
        raise KeyError(f"Study {name} not found")

    def register(self, study: StudyData | list[StudyData]) -> "StudyCollection":
        """Register a new study with the class"""
        study = [study] if isinstance(study, StudyData) else study

        for s in study:
            self.collection.append(s)

        return self

    def __len__(self):
        return len(self.collection)

    @property
    def df(self) -> pd.DataFrame:
        """Dataframe of available studies"""
        return (
            pd.concat([study.df for study in self.collection], ignore_index=True)
            if len(self.collection) > 0
            else pd.DataFrame()
        )

    @classmethod
    def from_yaml(cls, file_path: str) -> "StudyCollection":
        """Initialize class with studies from yaml file"""
        with Path.open(file_path, "r") as file:
            studies = yaml.safe_load(file.read())

        studies = [StudyData(**study) for study in studies]

        return cls().register(studies)

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks"""
        return self.df.to_html()


# Initialize class and register all available studies
study_collection = StudyCollection.from_yaml(STUDY_CONFIG)


def available_data() -> StudyCollection:
    """Get list all available studies

    Returns
    -------
    StudyCollection
        A study collection object that stores the metadata of all accessible studies

    Example
    -------

    .. code-block:: python

        alphatools.data.available_data()
        > Collection of 1 study
            - bader2020_pg_alphadia (pg, alphadia): Study on Cerebral Spinal Fluid of Alzheimer patients by Bader et al.

    """
    return study_collection


def get_data(study: str, output_dir: str | Path | None = None) -> Path:
    """Get data from a specific study"""
    study_data = study_collection.get_study(study)
    return Path(study_data.download(output_dir))
