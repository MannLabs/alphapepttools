import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphapepttools.tl.tools import get_id2gene_map, map_genes_to_protein_groups
from alphapepttools.tl.utils import (
    drop_features_with_too_few_valid_values,
    find_iterable_kwargs,
    validate_ttest_inputs,
)

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


def test_get_id2gene_map_invalid_source_type(example_fasta):
    """`source_type` outside {'file', 'string'} should raise ValueError."""
    with pytest.raises(ValueError, match="source_type must be either"):
        get_id2gene_map(example_fasta, source_type="invalid")


def test_get_id2gene_map_invalid_input_type():
    """Non-str/Path input should raise TypeError."""
    with pytest.raises(TypeError, match="fasta_input must be a Path or string"):
        get_id2gene_map(12345, source_type="string")


def test_map_genes_to_protein_groups_all_unmapped():
    """If no protein in a group has a gene name, the group should map to 'NA'."""
    result = map_genes_to_protein_groups(
        id2gene_map={"ID0": "GN0"},  # only ID0 mapped
        protein_groups=["ID_unmapped_1;ID_unmapped_2"],
        delimiter=";",
    )
    assert result == ["NA"]


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


def test_find_iterable_kwargs_invalid_type():
    """Non-dict input raises TypeError."""
    with pytest.raises(TypeError, match="dictionary of keyword arguments"):
        find_iterable_kwargs([1, 2, 3])


### Test validate_ttest_inputs ###


@pytest.fixture
def simple_ttest_adata():
    """AnnData with 5 samples of group A and 3 of group B (imbalanced for sample-count tests)."""
    return ad.AnnData(
        X=np.random.rand(8, 3),
        obs=pd.DataFrame(
            {"condition": ["A"] * 5 + ["B"] * 3},
            index=[f"s{i}" for i in range(8)],
        ),
    )


@pytest.mark.parametrize(
    ("between_column", "comparison", "min_valid_values", "match"),
    [
        # missing between_column
        ("nonexistent", ("A", "B"), 2, "not found in adata.obs"),
        # comparison is a list, not a tuple
        ("condition", ["A", "B"], 2, "tuple of exactly two"),
        # comparison has wrong length
        ("condition", ("A", "B", "C"), 2, "tuple of exactly two"),
        # g1 not in available groups
        ("condition", ("Z", "B"), 2, "Group 'Z' not found"),
        # g2 not in available groups
        ("condition", ("A", "Z"), 2, "Group 'Z' not found"),
        # g1 has too few samples (A=5 < 6)
        ("condition", ("A", "B"), 6, "Group 'A' has only"),
        # g2 has too few samples (g1 passes first: A=5≥4, then B=3 < 4)
        ("condition", ("A", "B"), 4, "Group 'B' has only"),
    ],
)
def test_validate_ttest_inputs_raises(simple_ttest_adata, between_column, comparison, min_valid_values, match):
    with pytest.raises(ValueError, match=match):
        validate_ttest_inputs(simple_ttest_adata, between_column, comparison, min_valid_values=min_valid_values)
