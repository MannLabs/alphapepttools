"""Tests for peptide-level aggregation functions."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def precursor_adata():
    """Create mock precursor-level AnnData for testing.

    Creates 4 precursors that aggregate to 2 peptides:
    - PEPTIDEK3 and PEPT(Mod1)IDEK2 � both strip to PEPTIDEK, protein P1;P1-1, gene G1
    - DIPEPTKEK2 and DIPE(Mod2)PTKEK3 � both strip to DIPEPTKEK, protein P2, gene G2

    Quantification values:
    - sample_1: [10, 100, 3, nan] � after max aggregation: [100, 3]
    - sample_2: [100, 30, nan, nan] � after max aggregation: [100, nan]
    """
    # Define precursor names (with charge states and modifications)
    precursor_names = ["PEPTIDEK3", "PEPT(Mod1)IDEK2", "DIPEPTKEK2", "DIPE(Mod2)PTKEK3"]

    # Quantification matrix: samples � precursors
    X = np.array(
        [
            [10.0, 100.0, 3.0, np.nan],  # sample_1
            [100.0, 30.0, np.nan, np.nan],  # sample_2
        ]
    )

    # Sample metadata (obs)
    obs = pd.DataFrame({"cell_type": ["T1", "T1"], "treatment": ["A", "B"]}, index=["sample_1", "sample_2"])

    # Precursor metadata (var)
    var = pd.DataFrame(
        {
            "sequence": ["PEPTIDEK", "PEPTIDEK", "DIPEPTKEK", "DIPEPTKEK"],
            "proteins": ["P1;P1-1", "P1;P1-1", "P2", "P2"],
            "genes": ["G1", "G1", "G2", "G2"],
        },
        index=precursor_names,
    )

    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def expected_peptide_adata():
    """Create expected peptide-level AnnData after aggregation.

    Expected result after max aggregation:
    - 2 peptides: DIPEPTKEK_IN_P2, PEPTIDEK_IN_P1;P1-1 (alphabetical order)
    - sample_1: [3, 100] (max of [3] for DIPEPTKEK, max of [10, 100] for PEPTIDEK)
    - sample_2: [nan, 100] (nan for DIPEPTKEK, max of [100, 30] for PEPTIDEK)
    """
    # Expected index format: sequence_IN_protein (alphabetical order)
    peptide_indices = ["DIPEPTKEK_IN_P2", "PEPTIDEK_IN_P1;P1-1"]

    # Quantification matrix after max aggregation (matching alphabetical order)
    X = np.array(
        [
            [3.0, 100.0],  # sample_1: DIPEPTKEK then PEPTIDEK
            [np.nan, 100.0],  # sample_2: DIPEPTKEK then PEPTIDEK
        ]
    )

    # Sample metadata (unchanged from precursor level)
    obs = pd.DataFrame({"cell_type": ["T1", "T1"], "treatment": ["A", "B"]}, index=["sample_1", "sample_2"])

    # Peptide metadata (matching alphabetical order)
    var = pd.DataFrame(
        {"genes": ["G2", "G1"], "sequence": ["DIPEPTKEK", "PEPTIDEK"], "proteins": ["P2", "P1;P1-1"]},
        index=peptide_indices,
    )

    return ad.AnnData(X=X, obs=obs, var=var)


def test_group_peptides_basic(precursor_adata, expected_peptide_adata):
    """Test basic peptide grouping with max aggregation."""
    from alphapepttools.io.peptide_level import group_peptides

    result = group_peptides(
        precursor_adata,
        sequence_column="sequence",
        protein_id_column="proteins",
        aggregation="max",
        added_columns=["genes"],
    )

    # Check X matrix equality (using allclose for NaN handling)
    np.testing.assert_array_equal(result.X, expected_peptide_adata.X)

    # Check obs dataframe equality
    pd.testing.assert_frame_equal(result.obs, expected_peptide_adata.obs)

    # Check var dataframe equality (sorting columns to ensure order matches)
    result_var_sorted = result.var.sort_index(axis=1)
    expected_var_sorted = expected_peptide_adata.var.sort_index(axis=1)
    pd.testing.assert_frame_equal(result_var_sorted, expected_var_sorted)


def test_group_peptides_without_added_columns(precursor_adata):
    """Test peptide grouping without additional metadata columns."""
    from alphapepttools.io.peptide_level import group_peptides

    result = group_peptides(
        precursor_adata, sequence_column="sequence", protein_id_column="proteins", aggregation="max", added_columns=None
    )

    # Should still have sequence and proteins columns
    assert "sequence" in result.var.columns
    assert "proteins" in result.var.columns

    # Should NOT have genes column
    assert "genes" not in result.var.columns

    # Check dimensions
    assert result.n_obs == 2  # noqa: PLR2004
    assert result.n_vars == 2  # noqa: PLR2004


@pytest.fixture
def expected_peptide_adata_median():
    """Create expected peptide-level AnnData after median aggregation.

    Expected result after median aggregation:
    - 2 peptides: DIPEPTKEK_IN_P2, PEPTIDEK_IN_P1;P1-1 (alphabetical order)
    - sample_1: [3, 55] (median of [3] for DIPEPTKEK, median of [10, 100] for PEPTIDEK)
    - sample_2: [nan, 65] (nan for DIPEPTKEK, median of [100, 30] for PEPTIDEK)
    """
    # Expected index format: sequence_IN_protein (alphabetical order)
    peptide_indices = ["DIPEPTKEK_IN_P2", "PEPTIDEK_IN_P1;P1-1"]

    # Quantification matrix after median aggregation (matching alphabetical order)
    X = np.array(
        [
            [3.0, 55.0],  # sample_1: DIPEPTKEK then PEPTIDEK
            [np.nan, 65.0],  # sample_2: DIPEPTKEK then PEPTIDEK
        ]
    )

    # Sample metadata (unchanged from precursor level)
    obs = pd.DataFrame({"cell_type": ["T1", "T1"], "treatment": ["A", "B"]}, index=["sample_1", "sample_2"])

    # Peptide metadata (matching alphabetical order)
    var = pd.DataFrame(
        {"genes": ["G2", "G1"], "sequence": ["DIPEPTKEK", "PEPTIDEK"], "proteins": ["P2", "P1;P1-1"]},
        index=peptide_indices,
    )

    return ad.AnnData(X=X, obs=obs, var=var)


def test_group_peptides_median_aggregation(precursor_adata, expected_peptide_adata_median):
    """Test peptide grouping with median aggregation."""
    from alphapepttools.io.peptide_level import group_peptides

    result = group_peptides(
        precursor_adata,
        sequence_column="sequence",
        protein_id_column="proteins",
        aggregation="median",
        added_columns=["genes"],
    )

    # Check X matrix equality
    np.testing.assert_array_equal(result.X, expected_peptide_adata_median.X)

    # Check obs dataframe equality
    pd.testing.assert_frame_equal(result.obs, expected_peptide_adata_median.obs)

    # Check var dataframe equality (sorting columns to ensure order matches)
    result_var_sorted = result.var.sort_index(axis=1)
    expected_var_sorted = expected_peptide_adata_median.var.sort_index(axis=1)
    pd.testing.assert_frame_equal(result_var_sorted, expected_var_sorted)


def test_group_peptides_invalid_added_columns(precursor_adata):
    """Test that invalid added_columns type raises TypeError."""
    from alphapepttools.io.peptide_level import group_peptides

    with pytest.raises(TypeError, match="added_columns must be a list"):
        group_peptides(
            precursor_adata,
            added_columns="genes",  # Should be a list, not a string
        )


def test_group_peptides_preserves_index_format(precursor_adata):
    """Test that output index follows expected format: sequence_IN_protein."""
    from alphapepttools.io.peptide_level import group_peptides

    result = group_peptides(
        precursor_adata,
        sequence_column="sequence",
        protein_id_column="proteins",
        aggregation="max",
        added_columns=["genes"],
    )

    # Check index format
    expected_indices = ["PEPTIDEK_IN_P1;P1-1", "DIPEPTKEK_IN_P2"]

    for expected_idx in expected_indices:
        assert expected_idx in result.var_names, f"Expected index {expected_idx} not found"
