import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.tl.tools import get_id2gene_map, map_genes2pg, scanpy_pycombat


@pytest.fixture
def pycombat_test_data():
    def make_dummy_data():
        df = pd.DataFrame(
            {"A": [1, 2, 4, 8, 16, 128, 256, 512, 1024, np.nan], "B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            index=list("ABCDEFGHIJ"),
        )
        md = pd.DataFrame({"batch1": list("xxxxxyyyyy"), "batch2": list("xxxxxxxxxy")}, index=list("ABCDEFGHIJ"))
        return df, md

    return make_dummy_data()


# Note: This test only ascertains that pyCombat is applied correctly as per previous implementations that produced
# biologically plausible results. The test does not assert the statistical correctness of the method itself.
@pytest.mark.parametrize(
    ("expected_data", "batch", "imputer", "return_imputed"),
    [
        # Basic case: gaussian imputation on adequately sized batches without returning NaN values
        (
            pd.DataFrame(
                {
                    "A": [
                        159.74269124980458,
                        161.7980312113076,
                        165.90871113431353,
                        174.13007098032546,
                        190.5727906723493,
                        -46.13527170443727,
                        45.68352928782083,
                        229.3211312723371,
                        596.5963352413695,
                        np.nan,
                    ],
                    "B": [
                        3.0966256063595146,
                        4.162899075334143,
                        5.2291725443087715,
                        6.295446013283399,
                        7.361719482258028,
                        4.236674696623119,
                        5.021086375704235,
                        5.805498054785351,
                        6.5899097338664685,
                        7.374321412947585,
                    ],
                },
                index=list("ABCDEFGHIJ"),
            ),
            "batch1",
            "gaussian",
            False,
        ),
        # Case with returning imputed values
        (
            pd.DataFrame(
                {
                    "A": [
                        159.74269124980458,
                        161.7980312113076,
                        165.90871113431353,
                        174.13007098032546,
                        190.5727906723493,
                        -46.13527170443727,
                        45.68352928782083,
                        229.3211312723371,
                        596.5963352413695,
                        -667.4331118200473,
                    ],
                    "B": [
                        3.0966256063595146,
                        4.162899075334143,
                        5.2291725443087715,
                        6.295446013283399,
                        7.361719482258028,
                        4.236674696623119,
                        5.021086375704235,
                        5.805498054785351,
                        6.5899097338664685,
                        7.374321412947585,
                    ],
                },
                index=list("ABCDEFGHIJ"),
            ),
            "batch1",
            "gaussian",
            True,
        ),
        # Case with undersized batch (only 1 sample), causing that sample to be dropped
        (
            pd.DataFrame(
                {
                    "A": [
                        13.340537916403633,
                        14.283346957985742,
                        16.168965041149846,
                        19.940201207478083,
                        27.482673540134584,
                        133.0772861973257,
                        253.75684351982983,
                        495.11595816483805,
                        977.8341874548544,
                    ],
                    "B": [
                        1.2287638336717461,
                        2.1715728752538097,
                        3.114381916835873,
                        4.057190958417936,
                        5.0,
                        5.942809041582064,
                        6.885618083164127,
                        7.82842712474619,
                        8.771236166328254,
                    ],
                },
                index=list("ABCDEFGHI"),
            ),
            "batch2",
            "gaussian",
            False,
        ),
    ],
)
def test_scanpy_pycombat(pycombat_test_data, expected_data, batch, imputer, return_imputed):
    """
    Test the scanpy_pycombat function with various scenarios.
    """
    df, md = pycombat_test_data
    df = df.astype(np.float64)  # Ensure the DataFrame is of float type to avoid anndata warnings
    adata = ad.AnnData(df, obs=md)
    adata = scanpy_pycombat(adata, batch=batch, imputer=imputer, return_imputed=return_imputed)

    pd.testing.assert_frame_equal(adata.to_df(), expected_data)


# Test the get_id2gene_map function
@pytest.fixture
def example_fasta():
    def make_dummy_data():
        return """>tr|ID0|ID0_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN0 PE=1 SV=1
PEPTIDEKPEPTIDEK
>tr|ID1|ID1_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN1 PE=1 SV=1
PEPTIDEKPEPTIDEK
>tr|ID2|ID2_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN1 PE=1 SV=1
PEPTIDEKPEPTIDEK
>tr|ID3|ID3_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN3 PE=1 SV=1
PEPTIDEKPEPTIDEK
>tr|ID4|ID4_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN4 PE=1 SV=1
PEPTIDEKPEPTIDEK"""

    return make_dummy_data()


@pytest.mark.parametrize(
    ("expected_dict"),
    [
        {"ID0": "GN0", "ID1": "GN1", "ID2": "GN1", "ID3": "GN3", "ID4": "GN4"},
    ],
)
def test_get_id2gene_map(example_fasta, expected_dict):
    id2gene = get_id2gene_map(example_fasta)

    assert id2gene == expected_dict


# Test the map_genes2pg function
@pytest.fixture
def example_protein_groups():
    def make_dummy_data():
        return ["ID0", "ID1;ID2", "ID3;ID4"]

    return make_dummy_data()


@pytest.mark.parametrize(
    ("id2gene", "expected_genes"),
    [({"ID0": "GN0", "ID1": "GN1", "ID2": "GN1", "ID3": "GN3", "ID4": "GN4"}, ["GN0", "GN1", "GN3;GN4"])],
)
def test_map_genes2pg(example_protein_groups, id2gene, expected_genes):
    mapped_genes = map_genes2pg(id2gene=id2gene, protein_groups=example_protein_groups, delimiter=";")

    assert mapped_genes == expected_genes
