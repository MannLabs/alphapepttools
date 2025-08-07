# Tools for data processing

import logging
from io import StringIO
from pathlib import Path

import anndata as ad
import numpy as np
import regex as re
import scanpy
from Bio import SeqIO

from alphatools.pp.impute import impute_gaussian

# logging configuration
logging.basicConfig(level=logging.INFO)


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
        logging.info(f"Reading FASTA from file path: {fasta_input!s}")
        with Path(fasta_input).open() as handle:
            fasta_data = list(SeqIO.parse(handle, "fasta"))
    else:
        logging.info("Parsing FASTA from string content")
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


def scanpy_pycombat(
    adata: ad.AnnData,
    batch: str,
    imputer: str = "gaussian",
    *,
    return_imputed: bool = False,
) -> ad.AnnData:
    """Apply batch correction using scanpy's implementation of pyComBat

    As a compromise between data missingness and not reporting imputed values, values are imputed if missing,
    but the imputed values are not returned as part of the result. This can be done optionally.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix, where rows are cells and columns are features.
    batch : str
        Name of the batch feature in obs, the variation associated with this feature will be corrected
    imputer : str
        Method to impute missing values. Current options are: 'gaussian'.
    return_imputed : bool
        Return data with imputed values or set imputed values back to nan, default to False

    Returns
    -------
    adata : anndata.AnnData
        Annotated data matrix with batch correction applied.
        If `return_imputed` is False, imputed values are set back to NaN.

    """
    logging.info(f" |-> Apply pyComBat to correct for {batch}")

    # always copy for now, implement inplace later if needed
    adata = adata.copy()

    # pycombat crashes when the input data contains NaNs
    # missing values are always imputed and optionally returned
    IMPUTE = False
    na_mask = np.isnan(adata.X)
    if np.isnan(adata.X).any():
        logging.info(
            f" |-> Data contains {np.isnan(adata.X).sum()} nans. Imputing values for calculations with {imputer}."
        )
        IMPUTE = True

        if imputer == "gaussian":
            adata = impute_gaussian(adata)
        else:
            raise ValueError(f"Imputer {imputer} not supported. Use 'gaussian'.")

    # 'batch' column must not contain NaNs. If it does, simply add a "NA" batch
    if adata.obs[batch].isna().sum() > 0:
        adata.obs[batch] = adata.obs[batch].fillna("NA")

    # If any level of the to-correct data has only one level, drop it
    batch_level_count = adata.obs.groupby(batch)[batch].transform("count")

    if any(batch_level_count == 1):
        logging.info(f"There are single-sample batches for {batch}. Dropping these samples.")
        adata = adata[batch_level_count > 1, :].copy()
        na_mask = na_mask[batch_level_count > 1, :]

    # batch correct; apparently pyCombat from scanpy sets everything to nan if there is a batch that contains only one cell
    # i.e. if a sample only occurs once per plate, everything fails and we get all nans
    # this is a known issue, but it is not clear how to fix this https://github.com/scverse/scanpy/issues/1175
    # for now, we will just catch the error and return the data without batch correction
    if any(adata.obs[batch].value_counts() == 1):
        logging.warning(
            f" |-> tools.py/scanpy_pyComBat: for batch correction: at least one batch of {batch} contains only one single sample. This causes scanpy.pp.combat to fail and return all nans, therefore batch correction can not be performed."
        )
        raise ValueError("At least one batch contains only one single sample.")
    scanpy.pp.combat(adata, key=batch, inplace=True)

    # return data with imputed values if requested
    if not return_imputed and IMPUTE:
        logging.info(" |-> Imputed values will not be returned.")
        adata.X[na_mask] = np.nan

    return adata
