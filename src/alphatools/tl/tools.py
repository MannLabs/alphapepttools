# Tools for data processing

import logging
from io import StringIO
from pathlib import Path

import numpy as np
import regex as re
from Bio import SeqIO

# logging configuration
logging.basicConfig(level=logging.INFO)


def umap() -> None:
    """Perform UMAP on the data"""
    raise NotImplementedError


def get_id2gene_map(
    fasta_input: str | Path,
    source_type: str = "file",
) -> dict[str, str]:
    """Reannotate protein groups with gene names from a FASTA input.

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

    if source_type not in ["file", "string"]:
        raise ValueError("source_type must be either 'file' or 'string'.")

    if not isinstance(fasta_input, str | Path):
        raise TypeError("fasta_input must be a Path or string.")

    if source_type == "file":
        logging.info(f"Reading FASTA from file path: {fasta_input!s}")
        # Context manager for path and string input
        if isinstance(fasta_input, Path):
            with fasta_input.open() as handle:
                # Iterator to list while file is open
                fasta_data = list(SeqIO.parse(handle, "fasta"))
        else:
            with Path(fasta_input).open() as handle:
                fasta_data = list(SeqIO.parse(handle, "fasta"))
    else:
        logging.info("Parsing FASTA from string content")
        with StringIO(fasta_input) as handle:
            fasta_data = list(SeqIO.parse(handle, "fasta"))

    pattern = re.compile(r"GN=([^\s]+)")
    for record in fasta_data:
        uniprot_id = record.id.split("|")[1]

        match = re.search(pattern, record.description)
        gene_name = match.group(1) if match else uniprot_id
        id2gene[uniprot_id] = gene_name

    return id2gene


def map_genes2pg(
    id2gene: dict,
    protein_groups: list[str],
    delimiter: str = ";",
) -> list[str]:
    """Map gene names to protein groups based

    Protein groups may consist of multiple UniProt IDs, separated by a delimiter.
    This function maps iterates each protein group and assigns the corresponding unique
    genes to the protein group.

    Parameters
    ----------
    id2gene : dict
        Dictionary mapping UniProt IDs to gene names
    id_column : list
        List containing protein group identifiers, where each identifier may consist of multiple UniProt IDs
    delimiter : str, optional
        Delimiter used to separate UniProt IDs in the protein group identifiers, by default ";"

    Returns
    -------
    list
        List of gene names corresponding to each protein group identifier.
        If no gene name could be found, "NA" is returned.

    """
    out_gene_names = []
    for pg in protein_groups:
        gene_names = [id2gene.get(p, "NA") for p in pg.split(delimiter)]

        if list(set(gene_names)) == ["NA"]:
            gene_names = ["NA"]
        else:
            gene_names = [g for g in gene_names if g != "NA"]
            gene_names = list(np.unique(np.array(gene_names)))

        out_gene_names.append(";".join(gene_names))

    return out_gene_names
