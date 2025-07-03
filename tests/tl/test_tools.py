import pytest

from alphatools.tl.tools import get_id2gene_map, map_genes2pg

DUMMY_FASTA = """>tr|ID0|ID0_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN0 PE=1 SV=1
PEPTIDEKPEPTIDEK
>tr|ID1|ID1_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN1 PE=1 SV=1
PEPTIDEKPEPTIDEK
>tr|ID2|ID2_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN1 PE=1 SV=1
PEPTIDEKPEPTIDEK
>tr|ID3|ID3_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN3 PE=1 SV=1
PEPTIDEKPEPTIDEK
>tr|ID4|ID4_HUMAN Protein1 OS=Homo sapiens OX=9606 GN=GN4 PE=1 SV=1
PEPTIDEKPEPTIDEK"""


@pytest.fixture
def example_fasta():
    def make_dummy_data():
        return DUMMY_FASTA

    return make_dummy_data()


@pytest.fixture
def example_fasta_file_path(tmp_path):
    data = DUMMY_FASTA
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(data)
    return fasta_file


@pytest.fixture
def example_fasta_file_string(tmp_path):
    data = DUMMY_FASTA
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(data)
    return str(fasta_file)


# Test the get_id2gene_map function
@pytest.mark.parametrize(
    ("expected_dict", "source_type", "fasta_input"),
    [
        (
            {"ID0": "GN0", "ID1": "GN1", "ID2": "GN1", "ID3": "GN3", "ID4": "GN4"},
            "string",
            "example_fasta",
        ),
        (
            {"ID0": "GN0", "ID1": "GN1", "ID2": "GN1", "ID3": "GN3", "ID4": "GN4"},
            "file",
            "example_fasta_file_path",
        ),
        (
            {"ID0": "GN0", "ID1": "GN1", "ID2": "GN1", "ID3": "GN3", "ID4": "GN4"},
            "file",
            "example_fasta_file_string",
        ),
    ],
)
def test_get_id2gene_map(request, expected_dict, source_type, fasta_input):
    id2gene = get_id2gene_map(request.getfixturevalue(fasta_input), source_type=source_type)
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
