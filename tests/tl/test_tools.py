import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphapepttools.tl.tools import find_protease_cut_sites, get_id2gene_map, map_genes_to_protein_groups
from alphapepttools.tl.utils import drop_features_with_too_few_valid_values, find_iterable_kwargs

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
def test_map_genes_to_protein_groups(example_protein_groups, id2gene, expected_genes):
    mapped_genes = map_genes_to_protein_groups(
        id2gene_map=id2gene, protein_groups=example_protein_groups, delimiter=";"
    )

    assert mapped_genes == expected_genes


# Test ttest filtering function
@pytest.fixture
def example_adata():
    def make_dummy_data():
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
                "B": [5.0, np.nan, np.nan, 2.0, 1.0, 0.0],
                "C": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
                "D": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "E": [np.nan, np.nan, np.nan, 4.0, 5.0, 6.0],
            },
            index=["cell1", "cell2", "cell3", "cell4", "cell5", "cell6"],
        )

        return ad.AnnData(
            df.to_numpy().astype(np.float64),
            obs=pd.DataFrame(
                {
                    "cell": ["x", "x", "x", "y", "y", "y"],
                },
                index=["cell1", "cell2", "cell3", "cell4", "cell5", "cell6"],
            ),
            var=pd.DataFrame(
                {
                    "gene": ["a", "b", "c", "d", "e"],
                },
                index=["A", "B", "C", "D", "E"],
            ),
        )

    return make_dummy_data()


def test_drop_features_with_too_few_valid_values(example_adata):
    adata = example_adata
    filtered_adata = drop_features_with_too_few_valid_values(
        adata,
        between_column="cell",
        comparison=("x", "y"),
        min_valid_values=2,
    )

    expected_columns = ["A", "C"]
    assert list(filtered_adata.var_names) == expected_columns


# Test find_protease_cut_sites function
@pytest.mark.parametrize(
    ("sequences", "expected_counts"),
    [
        # Fully tryptic peptides (0 miscleavages)
        (["AAAAAA", "AAAAAAK", "AAAAAAAR"], [0, 0, 0]),
        # One miscleavage
        (["AAAAAKAAAAAR", "AAAAARAAAAAAK"], [1, 1]),
        # Two miscleavages
        (["AAAAAKAAAAAKAAAAAAR", "AAAAARAAAAARAAAAAAK"], [2, 2]),
        # Proline rule: KP and RP should NOT be counted as cleavage sites
        (["AAAAKPAAAAAR", "AAAARPAAAAAAK"], [0, 0]),
        # Mixed cases
        (["AAAAAA", "AAAAAAK", "AAAAAKAAAAAR", "AAAAAKAAAAAKAAAAAAR"], [0, 0, 1, 2]),
        # Edge case: empty sequence
        ([""], [0]),
        # Edge case: single residue
        (["K"], [0]),
    ],
)
def test_find_protease_cut_sites(sequences, expected_counts):
    adata = ad.AnnData(
        np.zeros((1, len(sequences))),
        var=pd.DataFrame({"sequence": sequences}),
    )

    counts = find_protease_cut_sites(adata, sequence_column="sequence")

    assert list(counts) == expected_counts


# Test the find_iterable_kwargs function
@pytest.fixture
def example_kwargs_with_iterables():
    """Fixture providing kwargs with various types including iterables."""
    return {
        "s": np.array([10, 20, 30, 40, 50]),  # NumPy array
        "alpha": 0.5,  # scalar
        "edgecolors": ["red", "blue", "green", "yellow", "purple"],  # list
        "linewidths": pd.Series([1, 2, 3, 4, 5]),  # pandas Series
        "marker": "o",  # string
        "label": "data",  # string
        "cmap": None,  # None
        "empty_list": [],  # empty list
        "single_item_list": [42],  # single item list
    }


@pytest.mark.parametrize(
    ("match_length", "expected_keys", "unexpected_keys"),
    [
        (
            None,  # No length filter
            ["s", "edgecolors", "linewidths", "empty_list", "single_item_list"],
            ["alpha", "marker", "label", "cmap"],
        ),
        (
            5,  # Filter to length 5
            ["s", "edgecolors", "linewidths"],
            ["empty_list", "single_item_list", "alpha", "marker"],
        ),
        (
            1,  # Filter to length 1
            ["single_item_list"],
            ["s", "edgecolors", "linewidths", "empty_list"],
        ),
        (
            0,  # Filter to length 0
            ["empty_list"],
            ["s", "edgecolors", "linewidths", "single_item_list"],
        ),
    ],
)
def test_find_iterable_kwargs_filtering(example_kwargs_with_iterables, match_length, expected_keys, unexpected_keys):
    """Test find_iterable_kwargs with different length filters."""
    result = find_iterable_kwargs(example_kwargs_with_iterables, match_length=match_length)

    # Check expected keys are present
    for key in expected_keys:
        assert key in result, f"Expected {key} to be in result"
        # Verify the values match the original
        original = example_kwargs_with_iterables[key]
        if isinstance(original, (np.ndarray, pd.Series)):
            assert np.array_equal(result[key], original)
        else:
            assert result[key] == original

    # Check unexpected keys are absent
    for key in unexpected_keys:
        assert key not in result, f"Did not expect {key} to be in result"


@pytest.mark.parametrize(
    ("input_kwargs", "expected_result"),
    [
        ({}, {}),  # Empty input
        ({"a": 1, "b": "test", "c": None, "d": 3.14}, {}),  # No iterables
        ({"text": "hello", "values": [1, 2, 3]}, {"values": [1, 2, 3]}),  # String excluded
    ],
)
def test_find_iterable_kwargs_edge_cases(input_kwargs, expected_result):
    """Test find_iterable_kwargs with edge cases."""
    result = find_iterable_kwargs(input_kwargs)
    assert result == expected_result
