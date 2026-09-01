"""Test contractual agreement between alphaquant pipeline and alphapepttools"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphapepttools.tl import diff_exp_alphaquant
from alphapepttools.tl.diff_exp.alphaquant_wrapper import _HAS_ALPHAQUANT


@pytest.fixture
def test_data_dir():
    """Return tests/tl/tl_test_data"""
    return Path(__file__).parent.parent / "tl" / "tl_test_data"


@pytest.fixture
def test_data(test_data_dir: Path) -> tuple[ad.AnnData, pd.DataFrame]:
    """Prepare data for alphaquant run"""
    samplemap = pd.read_csv(test_data_dir / "samplemap_200.tsv", sep="\t")

    # As alphaquant performs its own quantification, the protein-level data is not used
    # by the function. However, the sample metadata is extracted from anndata, so the shape could
    # correspond to the sample metadata
    adata = ad.AnnData(
        X=pd.DataFrame(np.zeros(samplemap.shape[0]), index=samplemap["sample"], columns=["dummy"]),
        obs=samplemap.set_index("sample"),
    )

    report = pd.read_csv(test_data_dir / "example_dataset_mouse_sn_top20peptides.tsv", sep="\t")

    return adata, report


@pytest.fixture(autouse=True)
def validate_test_data(test_data):
    """Validate that test data preparation works"""
    adata, _ = test_data

    assert "condition" in adata.obs.columns
    assert all(condition in adata.obs["condition"].tolist() for condition in ("brain", "kidney"))


@pytest.mark.skipif(not _HAS_ALPHAQUANT, reason="alphaquant not installed")
def test_diff_exp_alphaquant__integration(test_data: tuple[ad.AnnData, pd.DataFrame]) -> None:
    """Integration test that checks contract between alphaquant and alphapepttools

    Check that the results contain the expected columns
    """
    adata, report = test_data

    expected_comparison = "brain_VS_kidney"

    # The three levels are stacked into a single frame, separated by the modality column
    expected_columns = (
        "modality",
        "feature",
        "condition_pair",
        "protein",
        "log2fc",
        "p_value",
        "-log10(p_value)",
        "fdr",
        "-log10(fdr)",
        "method",
        "max_level_1_samples",
        "max_level_2_samples",
        "quality_score",
        "proteoform_id",
        "peptides",
        "num_peptides",
        "sequence",
    )
    # The report contains 7 proteins, 7 proteoforms and 20 peptides
    expected_row_counts = {"protein": 7, "proteoform": 7, "peptide": 20}

    results = diff_exp_alphaquant(
        adata=adata,
        report=report,
        between_column="condition",
        comparison=("brain", "kidney"),
        min_valid_values=2,
        valid_values_filter_mode="either",
        plots="hide",
    )

    assert tuple(results.columns) == expected_columns
    assert results.shape == (sum(expected_row_counts.values()), len(expected_columns))
    assert set(results["modality"].unique()) == set(expected_row_counts.keys())

    for feature_level, expected_n_rows in expected_row_counts.items():
        assert (results["modality"] == feature_level).sum() == expected_n_rows

    # Sanity checks - the other columns are not tested due to potential changes in the alphaquant backend
    assert results["feature"].notna().all()
    assert all(results["condition_pair"] == expected_comparison)
    assert all(results["method"] == "alphaquant")
