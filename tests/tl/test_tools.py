import pytest

from alphatools.tl.tools import get_id2gene_map, map_genes2pg


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


# TODO: Implement with actual fasta file reading from path as well
@pytest.mark.parametrize(
    ("expected_dict"),
    [
        {"ID0": "GN0", "ID1": "GN1", "ID2": "GN1", "ID3": "GN3", "ID4": "GN4"},
    ],
)
def test_get_id2gene_map(example_fasta, expected_dict):
    id2gene = get_id2gene_map(example_fasta, source_type="string")

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
