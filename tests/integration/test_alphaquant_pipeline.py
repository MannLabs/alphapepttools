"""Test contractual agreement between alphaquant pipeline and alphapepttools"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphapepttools.tl import diff_exp_alphaquant


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


def test_diff_exp_alphaquant__integration(test_data: tuple[ad.AnnData, pd.DataFrame]) -> None:
    """Integration test that checks contract between alphaquant and alphapepttools

    Check that the results contain the expected columns
    """
    adata, report = test_data

    expected_comparison = "brain_VS_kidney"
    expected_results = {
        "protein": {
            "columns": (
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
            ),
            # Report contains 7 proteins
            "shape": (7, 11),
        },
        "proteoform": {
            "columns": (
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
                "proteoform_id",
                "peptides",
                "num_peptides",
                "quality_score",
            ),
            # Report contains 7 proteoforms
            "shape": (7, 14),
        },
        "peptide": {
            "columns": (
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
                "sequence",
                "quality_score",
            ),
            # Report contains 20 peptides
            "shape": (20, 12),
        },
    }

    comparison_key, results = diff_exp_alphaquant(
        adata=adata,
        report=report,
        between_column="condition",
        comparison=("brain", "kidney"),
        min_valid_values=2,
        valid_values_filter_mode="either",
        plots="hide",
    )

    assert comparison_key == expected_comparison
    assert len(results) == len(expected_results)
    assert set(results.keys()) == set(expected_results.keys())

    for feature_level in expected_results:
        assert results[feature_level].shape == expected_results[feature_level]["shape"]
        assert tuple(results[feature_level].columns) == expected_results[feature_level]["columns"]

        # Sanity checks - the other columns are not tested due to potential changes in the alphaquant backend
        assert all(results[feature_level]["condition_pair"] == expected_comparison)
        assert all(results[feature_level]["method"] == "alphaquant")
