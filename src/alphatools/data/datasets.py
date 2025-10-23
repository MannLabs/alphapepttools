"""Example datasets"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from alphabase.tools.data_downloader import DataShareDownloader


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

    def __repr__(self):
        study_representation = [
            f"- {study.name} ({study.data_type}, {study.search_engine}): {study.description if study.description is not None else ''}"
            for study in self.collection
        ]

        study_representation = "\n\t".join(study_representation)

        return f"""Collection of {self.__len__()} {"study" if self.__len__() == 1 else "studies"}\n\t{study_representation}"""

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks"""
        if not self.collection:
            return "<p>Empty collection</p>"

        rows = ""
        for study in self.collection:
            description = study.description if study.description else ""
            citation = f"<small>{study.citation}</small>" if study.citation else ""
            rows += f"""
            <tr>
                <td><strong>{study.name}</strong></td>
                <td>{study.data_type}</td>
                <td>{study.search_engine}</td>
                <td>{description}</td>
                <td>{citation}</td>
            </tr>
            """

        html = f"""
        <div>
            <h3>Collection of {self.__len__} {"study" if self.__len__ == 1 else "studies"}</h3>
            <table style="border-collapse: collapse; width: 100%;">
                <thead>
                    <tr style="border-bottom: 2px solid #ddd;">
                        <th style="text-align: left; padding: 8px;">Name</th>
                        <th style="text-align: left; padding: 8px;">Data Type</th>
                        <th style="text-align: left; padding: 8px;">Search Engine</th>
                        <th style="text-align: left; padding: 8px;">Description</th>
                        <th style="text-align: left; padding: 8px;">Citation</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
        return html  # noqa: RET504 Remove unnecessary assignment - return value would be very long


bader2020_pg_alphadia = StudyData(
    name="bader2020_pg_alphadia",
    url="https://datashare.biochem.mpg.de/s/yLpjkoQzMHpdDsB",
    search_engine="alphadia",
    citation="Bader JM, Geyer PE, Müller JB, Strauss MT, Koch M, Leypoldt F, Koertvelyessy P, Bittner D, Schipke CG, Incesoy EI, Peters O, Deigendesch N, Simons M, Jensen MK, Zetterberg H, Mann M. Proteome profiling in cerebrospinal fluid reveals novel biomarkers of Alzheimer's disease. Mol Syst Biol. 2020 Jun;16(6):e9356. doi: 10.15252/msb.20199356. PMID: 32485097; PMCID: PMC7266499.",
    data_type="study_pg",
    description="Study on Cerebral Spinal Fluid of Alzheimer patients by Bader et al.",
)

bader2020_pg_diann = StudyData(
    name="bader2020_pg_diann",
    url="https://datashare.biochem.mpg.de/s/3oZsya2L5bGnmtQ",
    search_engine="DiaNN",
    citation="Bader JM, Geyer PE, Müller JB, Strauss MT, Koch M, Leypoldt F, Koertvelyessy P, Bittner D, Schipke CG, Incesoy EI, Peters O, Deigendesch N, Simons M, Jensen MK, Zetterberg H, Mann M. Proteome profiling in cerebrospinal fluid reveals novel biomarkers of Alzheimer's disease. Mol Syst Biol. 2020 Jun;16(6):e9356. doi: 10.15252/msb.20199356. PMID: 32485097; PMCID: PMC7266499.",
    data_type="study_pg",
    description="Study on Cerebral Spinal Fluid of Alzheimer patients by Bader et al.",
)

spectronaut_pg = StudyData(
    name="spectronaut_pg",
    url="https://datashare.biochem.mpg.de/s/Ai9TiBTeaPHK5by",
    search_engine="spectronaut",
    data_type="pg",
    description="An example spectronaut report",
)

bader2020_psm_alphadia = StudyData(
    name="bader2020_psm_alphadia",
    url="https://datashare.biochem.mpg.de/s/awYyxod4ksz86kk",
    search_engine="alphadia",
    citation="Bader JM, Geyer PE, Müller JB, Strauss MT, Koch M, Leypoldt F, Koertvelyessy P, Bittner D, Schipke CG, Incesoy EI, Peters O, Deigendesch N, Simons M, Jensen MK, Zetterberg H, Mann M. Proteome profiling in cerebrospinal fluid reveals novel biomarkers of Alzheimer's disease. Mol Syst Biol. 2020 Jun;16(6):e9356. doi: 10.15252/msb.20199356. PMID: 32485097; PMCID: PMC7266499.",
    data_type="study_psm",
    description="Study on Cerebral Spinal Fluid of Alzheimer patients by Bader et al.",
)

bader2020_psm_diann = StudyData(
    name="bader2020_psm_diann",
    url="https://datashare.biochem.mpg.de/s/c4Z5Yg6srKQyDym",
    search_engine="DiaNN",
    citation="Bader JM, Geyer PE, Müller JB, Strauss MT, Koch M, Leypoldt F, Koertvelyessy P, Bittner D, Schipke CG, Incesoy EI, Peters O, Deigendesch N, Simons M, Jensen MK, Zetterberg H, Mann M. Proteome profiling in cerebrospinal fluid reveals novel biomarkers of Alzheimer's disease. Mol Syst Biol. 2020 Jun;16(6):e9356. doi: 10.15252/msb.20199356. PMID: 32485097; PMCID: PMC7266499.",
    data_type="study_psm",
    description="Study on Cerebral Spinal Fluid of Alzheimer patients by Bader et al.",
)

spectronaut_psm = StudyData(
    name="spectronaut_psm",
    url="https://datashare.biochem.mpg.de/s/GtfJL49Rf9w78EE",
    search_engine="spectronaut",
    data_type="study_psm",
    description="An example spectronaut report",
)


# Initialize class and register all available studies
study_collection = StudyCollection()
study_collection.register(
    [
        bader2020_pg_alphadia,
        bader2020_pg_diann,
        spectronaut_pg,
        bader2020_psm_alphadia,
        bader2020_psm_diann,
        spectronaut_psm,
    ]
)


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
