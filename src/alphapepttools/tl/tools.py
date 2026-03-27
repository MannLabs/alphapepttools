# Tools for data processing

import logging
from collections.abc import Iterable
from io import StringIO
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import regex as re
from Bio import SeqIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_id2gene_map(
    fasta_input: str | Path,
    source_type: str = "file",
) -> dict[str, str]:
    r"""Reannotate protein groups with gene names from a FASTA input.

    The function tries to extract UniProt IDs from the second position in a standard fasta header (see example below),
    and match the gene name based on whatever comes after the `'GN='` tag in the header (matching via regex `r"GN=([^\s]+)"`).
    The fasta file typically corresponds to the file that was used during the search step.

    Parameters
    ----------
    fasta_input
        If source_type is 'file' (default), this is interpreted as a filepath to a FASTA file.
        If source_type is 'string', this is parsed directly as a string-format fasta (multi-line with headers and sequences)
    source_type
        Specifies the source type of the FASTA input, either 'file' or 'string'.
        Defaults to 'file'.

    Returns
    -------
    A dictionary mapping UniProt IDs to gene names. If no gene name is found,
    the UniProt ID is used as fallback.

    Examples
    --------
    Example for string FASTA input:

    .. code-block:: python

        fasta_string = '''\
        >tr|ID0|ID0_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN0 PE=1 SV=1
        PEPTIDEKPEPTIDEK
        >tr|ID1|ID1_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN1 PE=1 SV=1
        PEPTIDEKPEPTIDEK
        '''

        alphatools.tools.get_id2gene_map(fasta_string, source_type="string")
        > {'ID0': 'GN0', 'ID1': 'GN1'}
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
    protein_groups: Iterable[str],
    delimiter: str = ";",
) -> list[str]:
    r"""Map gene names to protein groups using the provided id2gene_map mapping

    Protein groups may consist of multiple UniProt IDs, separated by a delimiter.
    This function iterates over each protein group and assigns the corresponding unique
    genes to the protein group.

    Parameters
    ----------
    id2gene_map
        Dictionary mapping UniProt IDs to gene names
    protein_groups
        List containing protein group identifiers, where each identifier may consist of multiple UniProt IDs
    delimiter
        Delimiter used to separate UniProt IDs in the protein group identifiers, by default ";"

    Examples
    --------
    You can map a list of uniprot IDs to gene names

    .. code-block:: python

        id2gene_map = {"ID0": "GN0", "ID1": "GN1", "ID2": "GN1", "ID3": "GN3", "ID4": "GN4"}
        protein_groups = ["ID0", "ID1;ID2", "ID3;ID4"]
        map_genes_to_protein_groups(id2gene_map, protein_groups, delimiter=";")
        > ["GN0", "GN1", "GN3;GN4"]

    To map gene names to an AnnData object, you can use the :func:`get_id2gene_map` function
    to create a mapping from a FASTA file or string and subsequently assign the extracted gene
    names to the `adata.var` attribute

    .. code-block:: python

        from alphapepttools.tl.tools import get_id2gene_map, map_genes_to_protein_groups

        fasta = '''\
        >tr|ID0|ID0_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN0 PE=1 SV=1
        PEPTIDEKPEPTIDEK
        >tr|ID1|ID1_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN1 PE=1 SV=1
        PEPTIDEKPEPTIDEK
        '''
        mapping = get_id2gene_map(fasta, source_type="string")
        mapping
        # {'ID0': 'GN0', 'ID1': 'GN1'}

        adata.var
        # Empty DataFrame
        # Columns: []
        # Index: [ID0, ID1]

        adata.var["gene_id"] = map_genes_to_protein_groups(
            id2gene_map=mapping, protein_groups=adata.var_names
        )

    Returns
    -------
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


def find_protease_cut_sites(
    adata: ad.AnnData,
    sequence_column: str = "sequence",
    cleavage_pattern: str = r"(?<!P)[KR](?!P)",
) -> pd.Series:
    """Find internal protease cut sites in peptide sequences to detect miscleavages

    The cleavage pattern can be defined as a regex pattern, and looks for tryptic
    cleavage sites by default (K or R not followed by P). The function counts only
    internal cleavage sites by excluding the C-terminal residue from the search.

    Parameters
    ----------
    adata
        AnnData object containing peptide-level data. Must have `sequence_column` in `adata.var`.
    sequence_column
        Column name in `adata.var` containing peptide sequences. Defaults to "sequence".
    cleavage_pattern
        Regular expression pattern defining the protease cleavage sites. Defaults to r"(?<!P)[KR](?!P)" for trypsin.

    Returns
    -------
    pd.Series
        Series containing the count of internal cleavage sites for each peptide sequence.
        A value of 0 indicates no miscleavages (fully tryptic), values ≥1 indicate the number of missed cleavages.

    """
    if sequence_column not in adata.var.columns:
        raise ValueError(f"{sequence_column} column not found in adata.var.columns, is this a precursor table?")

    # replace empty sequences with a placeholder to avoid regex errors
    valid_sequences = adata.var[sequence_column].apply(lambda x: x + "_" if len(x) == 0 else x)

    # remove C-terminal residue to focus on internal cleavage sites
    valid_sequences_no_c_term = valid_sequences.apply(lambda x: x[:-1])

    return valid_sequences_no_c_term.apply(lambda x: len(re.findall(cleavage_pattern, x)))
