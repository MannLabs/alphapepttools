# Tools for data processing

import logging
from io import StringIO
from pathlib import Path

import numpy as np
import regex as re
from Bio import SeqIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def umap() -> None:
    """Perform UMAP on the data"""
    raise NotImplementedError


def get_id2gene_map(
    fasta_input: str | Path,
    source_type: str = "file",
) -> dict[str, str]:
    r"""Reannotate protein groups with gene names from a FASTA input.

    The function tries to extract UniProt IDs from the second position in a standard fasta header (see example below),
    and match the gene name based on whatever comes after the 'GN=' tag in the header (matching via regex r"GN=([^\s]+)").

    Parameters
    ----------
    fasta_input : str | Path
        If source_type is 'file' (default), this is interpreted as a filepath to a FASTA file.
        If source_type is 'string', this is parsed directly as a string-format fasta (multi-line with headers and sequences)
    source_type : str, optional
        Specifies the source type of the FASTA input, either 'file' or 'string'.
        Defaults to 'file'.

    Example for string FASTA input:
    ">tr|ID0|ID0_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN0 PE=1 SV=1
    PEPTIDEKPEPTIDEK
    >tr|ID1|ID1_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN1 PE=1 SV=1
    PEPTIDEKPEPTIDEK"

    Returns
    -------
    dict
        A dictionary mapping UniProt IDs to gene names. If no gene name is found,
        the UniProt ID is used as fallback.
    """
    id2gene = {}
    GENE_PATTERN = re.compile(r"GN=([^\s]+)")

    if source_type not in ["file", "string"]:
        raise ValueError("source_type must be either 'file' or 'string'.")

    if not isinstance(fasta_input, str | Path):
        raise TypeError("fasta_input must be a Path or string.")

    if source_type == "file":
        logger.info(f"Reading FASTA from file path: {fasta_input!s}")
        with Path(fasta_input).open() as handle:
            fasta_data = list(SeqIO.parse(handle, "fasta"))
    else:
        logger.info("Parsing FASTA from string content")
        with StringIO(fasta_input) as handle:
            fasta_data = list(SeqIO.parse(handle, "fasta"))

    for record in fasta_data:
        protein_id = record.id.split("|")[1]

        match = re.search(GENE_PATTERN, record.description)
        gene_name = match.group(1) if match else protein_id
        id2gene[protein_id] = gene_name

    return id2gene


def map_genes_to_protein_groups(
    id2gene_map: dict,
    protein_groups: list[str],
    delimiter: str = ";",
) -> list[str]:
    """Map gene names to protein groups based

    Protein groups may consist of multiple UniProt IDs, separated by a delimiter.
    This function maps iterates each protein group and assigns the corresponding unique
    genes to the protein group.

    Parameters
    ----------
    id2gene_map : dict
        Dictionary mapping UniProt IDs to gene names
    id_column : list
        List containing protein group identifiers, where each identifier may consist of multiple UniProt IDs
    delimiter : str, optional
        Delimiter used to separate UniProt IDs in the protein group identifiers, by default ";"

    Examples
    --------
    >>> id2gene_map = {"ID0": "GN0", "ID1": "GN1", "ID2": "GN1", "ID3": "GN3", "ID4": "GN4"}
    >>> protein_groups = ["ID0", "ID1;ID2", "ID3;ID4"]
    >>> map_genes2pg(id2gene_map, protein_groups, delimiter=";")
    ["GN0", "GN1", "GN3;GN4"]


    Returns
    -------
    list
        List of gene names corresponding to each protein group identifier.
        If no gene name could be found, "NA" is returned.

    """
    out_gene_names = []
    for protein_group in protein_groups:
        gene_names = [id2gene_map.get(protein, "NA") for protein in protein_group.split(delimiter)]

        if set(gene_names) == {"NA"}:
            gene_names = ["NA"]
        else:
            gene_names = [gene_name for gene_name in gene_names if gene_name != "NA"]
            gene_names = list(np.unique(np.array(gene_names)))

        out_gene_names.append(";".join(gene_names))

    return out_gene_names
