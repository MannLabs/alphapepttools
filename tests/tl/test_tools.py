import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphapepttools.tl.tools import get_id2gene_map, map_genes_to_protein_groups
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


# Test the find_iterable_kwargs function
def test_find_iterable_kwargs():
    # Test with various kwargs including arrays, scalars, and other types
    kwargs = {
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

    # Test without match_length (returns all iterables)
    result = find_iterable_kwargs(kwargs)
    assert "s" in result
    assert "edgecolors" in result
    assert "linewidths" in result
    assert "empty_list" in result
    assert "single_item_list" in result
    assert "alpha" not in result  # scalar
    assert "marker" not in result  # string
    assert "label" not in result  # string
    assert "cmap" not in result  # None
    assert np.array_equal(result["s"], kwargs["s"])
    assert result["edgecolors"] == kwargs["edgecolors"]
    assert np.array_equal(result["linewidths"], kwargs["linewidths"])

    # Test with match_length=5 (filters to matching length only)
    result_filtered = find_iterable_kwargs(kwargs, match_length=5)
    assert "s" in result_filtered
    assert "edgecolors" in result_filtered
    assert "linewidths" in result_filtered
    assert "empty_list" not in result_filtered  # wrong length
    assert "single_item_list" not in result_filtered  # wrong length
    assert len(result_filtered["s"]) == 5  # noqa: PLR2004
    assert len(result_filtered["edgecolors"]) == 5  # noqa: PLR2004
    assert len(result_filtered["linewidths"]) == 5  # noqa: PLR2004

    # Test with match_length=1
    result_single = find_iterable_kwargs(kwargs, match_length=1)
    assert "single_item_list" in result_single
    assert "s" not in result_single
    assert "edgecolors" not in result_single

    # Test with empty kwargs
    empty_result = find_iterable_kwargs({})
    assert empty_result == {}

    # Test with no iterables
    no_iterables = {"a": 1, "b": "test", "c": None, "d": 3.14}
    result_none = find_iterable_kwargs(no_iterables)
    assert result_none == {}

    # Test that strings are not considered iterables
    string_kwargs = {"text": "hello", "values": [1, 2, 3]}
    result_strings = find_iterable_kwargs(string_kwargs)
    assert "text" not in result_strings
    assert "values" in result_strings
