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
    """Container for proteomics study metadata and data access

    Represents a single proteomics study with its associated metadata, download
    capabilities, and data type information. Each study can be downloaded and
    loaded for analysis using alphapepttools.

    Parameters
    ----------
    name
        Unique study identifier used for retrieval
    url
        URL/datashare link where data is stored
    search_engine
        Name of search engine with which the data was searched (e.g., 'diann', 'maxquant', 'alphadia')
    data_type
        Type of data contained in the study:
            - 'pg': Protein group (PG) table
            - 'psm': Peptide spectrum match (PSM) table
            - 'study_psm': Full study data (metadata + PSM table)
            - 'study_pg': Full study data (metadata + PG table)
            - 'shapes': Shapes file (geojson)
            - 'image': Image file
            - 'anndata': AnnData object
    citation
        Citation reference for the study, optional
    description
        Human-readable description of the study content, optional

    Examples
    --------
    .. code-block:: python

        from alphapepttools.data.datasets import StudyData

        # Create a study data instance
        study = StudyData(
            name="my_experiment",
            url="https://datashare.com/my_data.parquet",
            search_engine="diann",
            data_type="pg",
            citation="Smith et al., 2024",
            description="Proteomics analysis of cell lines",
        )

        # Download the data
        file_path = study.download(output_dir="./data")

        # View study metadata
        print(study.df)

    """

    name: str
    url: str
    search_engine: str
    data_type: Literal["pg", "psm", "study_psm", "study_pg", "image", "shapes", "anndata"]
    citation: str | None = None
    description: str | None = None

    def download(self, output_dir: str | Path | None = None) -> Path:
        """Download study data to local directory

        Downloads the study data file from the configured URL to a local directory
        using the DataShareDownloader utility. If the file already exists locally,
        it may be returned without re-downloading (depending on downloader behavior).

        Parameters
        ----------
        output_dir
            Directory where the data should be downloaded. If None, uses current working directory

        Returns
        -------
        Path to the downloaded data file

        """
        output_dir = Path.cwd() if output_dir is None else output_dir
        return Path(DataShareDownloader(url=self.url, output_dir=output_dir).download())

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
    """Collection of proteomics studies with metadata and download capabilities

    Manages a collection of StudyData objects, providing methods to search, retrieve,
    and display available proteomics datasets. The collection can be initialized from
    YAML configuration files and provides a convenient interface for accessing
    multiple studies.

    Examples
    --------
    .. code-block:: python

        from alphapepttools.data.datasets import StudyCollection, StudyData

        # Create a new collection
        collection = StudyCollection()

        # Add studies to the collection
        study1 = StudyData(
            name="experiment1", url="https://data.url/file1.parquet", search_engine="diann", data_type="pg"
        )
        collection.register(study1)

        # Get a specific study
        study = collection.get_study("experiment1")

        # View all studies as DataFrame
        print(collection.df)

    """

    def __init__(self):
        self.collection: list[StudyData] = []

    def get_study(self, name: str) -> StudyData:
        """Retrieve a study by its unique name

        Searches the collection for a study with the specified name and returns
        the corresponding StudyData object.

        Parameters
        ----------
        name
            Unique identifier of the study to retrieve

        Returns
        -------
        StudyData object for the requested study

        Raises
        ------
        KeyError
            If no study with the given name exists in the collection

        """
        for study in self.collection:
            if study.name == name:
                return study
        raise KeyError(f"Study {name} not found")

    def register(self, study: StudyData | list[StudyData]) -> "StudyCollection":
        """Add one or more studies to the collection

        Registers new StudyData objects with the collection, making them available
        for retrieval and download. Supports adding single studies or lists of studies.

        Parameters
        ----------
        study
            Single StudyData object or list of StudyData objects to add

        Returns
        -------
        The StudyCollection instance (self) for method chaining

        """
        study = [study] if isinstance(study, StudyData) else study

        for s in study:
            self.collection.append(s)

        return self

    def __len__(self):
        return len(self.collection)

    @property
    def df(self) -> pd.DataFrame:
        """DataFrame containing metadata for all studies in the collection

        Creates a consolidated DataFrame with one row per study, containing
        all metadata fields (name, url, search_engine, data_type, citation,
        description). Returns an empty DataFrame if the collection is empty.

        Returns
        -------
        DataFrame with study metadata

        """
        return (
            pd.concat([study.df for study in self.collection], ignore_index=True)
            if len(self.collection) > 0
            else pd.DataFrame()
        )

    @classmethod
    def from_yaml(cls, file_path: str) -> "StudyCollection":
        """Create a StudyCollection from a YAML configuration file

        Loads study definitions from a YAML file and creates a populated
        StudyCollection. The YAML file should contain a list of study
        configurations with required fields matching StudyData parameters.

        Parameters
        ----------
        file_path
            Path to YAML file containing study configurations

        Returns
        -------
        New StudyCollection instance populated with studies from the YAML file

        Examples
        --------
        .. code-block:: python

            from alphapepttools.data.datasets import StudyCollection

            # Load studies from configuration
            collection = StudyCollection.from_yaml("alphapepttools/src/alphapepttools/data/datasets.yaml")

            # Access loaded studies
            print(f"Loaded {len(collection)} studies")
            print(collection.df)

        """
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
    """List all available proteomics studies

    Returns the global StudyCollection containing all pre-configured proteomics
    datasets available for download through alphapepttools. The collection provides
    metadata about each study including data type, search engine, and citations.

    Returns
    -------
    A StudyCollection object containing metadata of all accessible studies

    Examples
    --------
    View available studies:

    .. code-block:: python

        import alphapepttools as at

        # Get the collection of available studies
        studies = at.data.available_data()

        # Display as DataFrame
        print(studies.df)

        # Check number of available studies
        print(f"Total studies available: {len(studies)}")

    Access a specific study:

    .. code-block:: python

        import alphapepttools as at

        # Get collection and retrieve specific study
        studies = at.data.available_data()
        pelsa_study = studies.get_study("pelsa_report_diann")

        # Download the study data
        file_path = pelsa_study.download()

    Notes
    -----
    The available studies are configured in the 'datasets.yaml' file that ships
    with alphapepttools. This collection is loaded once at module import and
    cached for efficiency.

    See Also
    --------
    get_data : Simplified interface to download study data directly
    StudyCollection : The collection class with methods for study management

    """
    return study_collection


def get_data(study: str, output_dir: str | Path | None = None) -> Path:
    """Download data from a specific study

    Downloads proteomics data from predefined study datasets and returns the path
    to the downloaded file. The data can then be loaded using alphapepttools
    readers for further analysis.

    Parameters
    ----------
    study
        Name of the study to download. Use `available_data()` to see all available studies
    output_dir
        Directory where the data should be downloaded. If None, uses current working directory

    Returns
    -------
    Path to the downloaded data file

    Raises
    ------
    KeyError
        If the specified study name is not found in the collection

    Examples
    --------
    Download and load PELSA data:

    .. code-block:: python

        import tempfile
        import pandas as pd
        import alphapepttools as at

        # Download the data using the alphapepttools data module
        report_path = at.data.get_data("pelsa_report_diann", output_dir=tempfile.mkdtemp())

        # Get the full report as DataFrame
        full_report = pd.read_parquet(report_path)

        # Create AnnData object with protein level data
        adata_protein = at.io.read_psm_table(
            file_paths=report_path,
            search_engine="diann",
            level="proteins",
            var_columns=["genes"],
        )

    Download to specific directory:

    .. code-block:: python

        from pathlib import Path
        import alphapepttools as at

        # Download to a specific project folder
        data_dir = Path("./my_project/data")
        data_dir.mkdir(parents=True, exist_ok=True)

        file_path = at.data.get_data("bader2020_pg_alphadia", output_dir=data_dir)
        print(f"Data downloaded to: {file_path}")

    See Also
    --------
    available_data : List all available study datasets
    StudyData.download : Lower-level download method for individual studies

    """
    study_data = study_collection.get_study(study)
    return Path(study_data.download(output_dir))
