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
) -> dict:
    """Reannotate protein groups with gene names from a FASTA input.

    Parameters
    ----------
    fasta_input : str | Path
        - If a Path or file path string, it's interpreted as a file path.
        - If a plain FASTA string (multi-line with headers and sequences), it is parsed directly.

    Returns
    -------
    dict
        A dictionary mapping UniProt IDs to gene names. If no gene name is found,
        the UniProt ID is used as fallback.
    """
    id2gene = {}

    if isinstance(fasta_input, Path):
        logging.info(f"Reading FASTA from file path: {fasta_input}")
        handle = Path.open(fasta_input)
    elif isinstance(fasta_input, str):
        logging.info("Parsing FASTA from string content")
        handle = StringIO(fasta_input)
    else:
        raise TypeError("fasta_input must be a valid file path or FASTA string.")

    with handle:
        fasta_data = SeqIO.parse(handle, "fasta")
        for record in fasta_data:
            uniprot_id = record.id.split("|")[1]

            match = re.search(r"GN=([^\s]+)", record.description)
            gene_name = match.group(1) if match else uniprot_id
            id2gene[uniprot_id] = gene_name

    return id2gene


def map_genes2pg(
    id2gene: dict,
    protein_groups: list,
    delimiter: str = ";",
) -> list:
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
