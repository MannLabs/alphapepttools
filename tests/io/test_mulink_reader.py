"""Unit tests for mulink_from_anndatas."""

from __future__ import annotations

from typing import ClassVar

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from alphapepttools.io.mulink_reader import mulink_from_anndatas

# Small hierarchy mirroring the tutorial example (TF / SERPINA3 / ALB).
# genes -> proteins -> peptides -> precursors, coarsest to finest.
GENES = ["TF", "SERPINA3", "ALB"]
PROTEINS_TO_GENES = {
    "P02787": "TF",
    "P01011": "SERPINA3",
    "P02768": "ALB",
}
PEPTIDES_TO_PROTEINS = {
    "PEPTF1": "P02787",
    "PEPTF2": "P02787",
    "PEPSER1": "P01011",
    "PEPALB1": "P02768",
}
PRECURSORS_TO_PEPTIDES = {
    "PEPTF1_2": "PEPTF1",
    "PEPTF1_3": "PEPTF1",
    "PEPTF2_2": "PEPTF2",
    "PEPSER1_2": "PEPSER1",
    "PEPALB1_2": "PEPALB1",
}
SAMPLES = ["sample1", "sample2", "sample3"]


def _make_adata(var: pd.DataFrame, obs: pd.DataFrame, *, seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.random((len(obs), len(var)))
    return ad.AnnData(X=X, obs=obs.copy(), var=var.copy())


@pytest.fixture
def anndatas() -> dict[str, ad.AnnData]:
    """Four-level hierarchy of AnnData objects with realistic var index names and link columns."""
    obs = pd.DataFrame(
        {"condition": ["control", "control", "treatment"]},
        index=pd.Index(SAMPLES, name="sample"),
    )

    gene_var = pd.DataFrame(
        {"description": [f"desc-{g}" for g in GENES]},
        index=pd.Index(GENES, name="genes"),
    )
    protein_var = pd.DataFrame(
        {
            "genes": list(PROTEINS_TO_GENES.values()),
            "review": ["reviewed"] * len(PROTEINS_TO_GENES),
        },
        index=pd.Index(list(PROTEINS_TO_GENES.keys()), name="proteins"),
    )
    peptide_var = pd.DataFrame(
        {
            "proteins": list(PEPTIDES_TO_PROTEINS.values()),
            "length": [len(p) for p in PEPTIDES_TO_PROTEINS],
        },
        index=pd.Index(list(PEPTIDES_TO_PROTEINS.keys()), name="sequence"),
    )
    precursor_var = pd.DataFrame(
        {
            "sequence": list(PRECURSORS_TO_PEPTIDES.values()),
            "charge": [2, 3, 2, 2, 2],
        },
        index=pd.Index(list(PRECURSORS_TO_PEPTIDES.keys()), name="precursor_id"),
    )

    return {
        "genes": _make_adata(gene_var, obs, seed=0),
        "proteins": _make_adata(protein_var, obs, seed=1),
        "peptides": _make_adata(peptide_var, obs, seed=2),
        "precursors": _make_adata(precursor_var, obs, seed=3),
    }


class TestRoundTrip:
    """The MuData wrapper must register every modality and surface each AnnData unchanged."""

    def test_returns_mudata(self, anndatas):
        mdata = mulink_from_anndatas(anndatas)
        assert isinstance(mdata, md.MuData)

    def test_registers_all_modalities(self, anndatas):
        mdata = mulink_from_anndatas(anndatas)
        assert set(mdata.mod.keys()) == set(anndatas.keys())

    @pytest.mark.parametrize("level", ["genes", "proteins", "peptides", "precursors"])
    def test_modality_var_preserved(self, anndatas, level):
        mdata = mulink_from_anndatas(anndatas)
        pd.testing.assert_frame_equal(mdata.mod[level].var, anndatas[level].var)

    @pytest.mark.parametrize("level", ["genes", "proteins", "peptides", "precursors"])
    def test_modality_obs_preserved(self, anndatas, level):
        mdata = mulink_from_anndatas(anndatas)
        pd.testing.assert_frame_equal(mdata.mod[level].obs, anndatas[level].obs)

    @pytest.mark.parametrize("level", ["genes", "proteins", "peptides", "precursors"])
    def test_modality_x_preserved(self, anndatas, level):
        mdata = mulink_from_anndatas(anndatas)
        np.testing.assert_array_equal(mdata.mod[level].X, anndatas[level].X)

    def test_link_attached_to_varp(self, anndatas):
        mdata = mulink_from_anndatas(anndatas)
        assert "feature_mapping" in mdata.varp
        n = len(mdata.var_names)
        assert mdata.varp["feature_mapping"].shape == (n, n)


class TestAncestorsQuery:
    """`mdata.link.query.ancestors(...)` returns a MuData filtered to the queried features and all
    finer-level features that link to them, with each modality's obs/var intact.
    """

    QUERY: ClassVar[list[str]] = ["TF", "SERPINA3"]
    EXPECTED: ClassVar[dict[str, set[str]]] = {
        "genes": {"TF", "SERPINA3"},
        "proteins": {"P02787", "P01011"},
        "peptides": {"PEPTF1", "PEPTF2", "PEPSER1"},
        "precursors": {"PEPTF1_2", "PEPTF1_3", "PEPTF2_2", "PEPSER1_2"},
    }

    @pytest.fixture
    def filtered(self, anndatas):
        mdata = mulink_from_anndatas(anndatas)
        return mdata.link.query.ancestors(self.QUERY)

    @pytest.mark.parametrize("level", ["genes", "proteins", "peptides", "precursors"])
    def test_features_filtered_to_expected(self, filtered, level):
        assert set(filtered.mod[level].var_names) == self.EXPECTED[level]

    @pytest.mark.parametrize("level", ["genes", "proteins", "peptides", "precursors"])
    def test_filtered_var_is_row_subset_of_original(self, anndatas, filtered, level):
        original_var = anndatas[level].var
        filtered_var = filtered.mod[level].var
        # all rows in filtered are present in original (by index) and unchanged
        pd.testing.assert_frame_equal(filtered_var, original_var.loc[filtered_var.index])

    @pytest.mark.parametrize("level", ["genes", "proteins", "peptides", "precursors"])
    def test_filtered_obs_preserved(self, anndatas, filtered, level):
        pd.testing.assert_frame_equal(filtered.mod[level].obs, anndatas[level].obs)

    @pytest.mark.parametrize("level", ["genes", "proteins", "peptides", "precursors"])
    def test_filtered_x_is_column_subset_of_original(self, anndatas, filtered, level):
        original = anndatas[level]
        filtered_mod = filtered.mod[level]
        positions = original.var_names.get_indexer(filtered_mod.var_names)
        np.testing.assert_array_equal(filtered_mod.X, original.X[:, positions])


class TestValidation:
    def test_rejects_unknown_feature_level(self, anndatas):
        anndatas["bogus"] = anndatas["genes"]
        with pytest.raises(ValueError, match="Invalid feature level"):
            mulink_from_anndatas(anndatas)

    def test_rejects_wrong_index_name(self, anndatas):
        anndatas["genes"].var.index.name = "wrong_name"
        with pytest.raises(ValueError, match="index name"):
            mulink_from_anndatas(anndatas)

    def test_rejects_missing_coarser_link_column(self, anndatas):
        anndatas["proteins"].var = anndatas["proteins"].var.drop(columns=["genes"])
        with pytest.raises(ValueError, match="missing column"):
            mulink_from_anndatas(anndatas)
