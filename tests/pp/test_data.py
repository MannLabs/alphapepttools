import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import alphapepttools as apt
from alphapepttools.pp.data import (
    _count_or_fraction_missing,
    _handle_overlapping_columns,
    _resolve_max_missing,
    _to_anndata,
    coerce_to_dataframe,
    data_column_to_array,
)


# example data
@pytest.fixture
def example_data():
    def make_dummy_data():
        X = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [7, 8, 9],
            }
        )
        X.index = ["cell1", "cell2", "cell3"]
        X.columns = ["G1", "G2", "G3"]
        return X

    return make_dummy_data()


# example sample metadata: one more sample than data
@pytest.fixture
def example_sample_metadata():
    def make_dummy_data():
        sample_metadata = pd.DataFrame({"cell_type": ["A", "C", "B", "D"], "batch": ["1", "3", "2", "4"]})
        sample_metadata.index = ["cell1", "cell3", "cell2", "cell4"]
        return sample_metadata

    return make_dummy_data()


# example feature metadata: one more feature than data
@pytest.fixture
def example_feature_metadata():
    def make_dummy_data():
        feature_metadata = pd.DataFrame(
            {
                "gene_name": ["gene1", "gene3", "gene2", "gene4"],
                "UniProtID": ["P12345", "P34567", "P23456", "P45678"],
            }
        )
        feature_metadata.index = ["G1", "G3", "G2", "G4"]
        return feature_metadata

    return make_dummy_data()


# example AnnData object for downstream tests
@pytest.fixture
def example_anndata():
    def make_dummy_data():
        adata = apt.pp.data._to_anndata(example_data())
        apt.pp.add_metadata(adata, example_sample_metadata(), axis=0)
        apt.pp.add_metadata(adata, example_feature_metadata(), axis=1)
        return adata

    return make_dummy_data()


### Tests ###


# test adding metadata to data
@pytest.mark.parametrize(
    (
        "expected_data",
        "expected_sample_metadata",
        "expected_feature_metadata",
        "metadata_size",  # 1 = incoming metadata has more keys than existing metadata; 0 = existing metadata has more keys than incoming metadata.
        "keep_data_shape",  # 1 = keep data shape, i.e. pad incoming metadata if it is missing keys; 0 = subset entire adata object to incoming keys.
        "keep_existing_metadata",  # 1 = append incoming metadata to existing metadata; 0 = overwrite existing metadata with incoming metadata.
    ),
    [
        # 1.: 100
        (
            pd.DataFrame(
                {"G1": [1, 2, 3], "G2": [4, 5, 6], "G3": [7, 8, 9]},
                index=["cell1", "cell2", "cell3"],
            ),
            pd.DataFrame(
                {"cell_type": ["A", "B", "C"], "batch": ["1", "2", "3"]},
                index=["cell1", "cell2", "cell3"],
            ),
            pd.DataFrame(
                {
                    "gene_name": ["gene1", "gene2", "gene3"],
                    "UniProtID": ["P12345", "P23456", "P34567"],
                },
                index=["G1", "G2", "G3"],
            ),
            1,
            0,
            0,
        ),
        # 2.: 110
        (
            pd.DataFrame(
                {"G1": [1, 2, 3], "G2": [4, 5, 6], "G3": [7, 8, 9]},
                index=["cell1", "cell2", "cell3"],
            ),
            pd.DataFrame(
                {"cell_type": ["A", "B", "C"], "batch": ["1", "2", "3"]},
                index=["cell1", "cell2", "cell3"],
            ),
            pd.DataFrame(
                {
                    "gene_name": ["gene1", "gene2", "gene3"],
                    "UniProtID": ["P12345", "P23456", "P34567"],
                },
                index=["G1", "G2", "G3"],
            ),
            1,
            1,
            0,
        ),
        # 3.: 101
        (
            pd.DataFrame(
                {"G1": [1, 2, 3], "G2": [4, 5, 6], "G3": [7, 8, 9]},
                index=["cell1", "cell2", "cell3"],
            ),
            pd.DataFrame(
                {"batch_new": ["11", "22", "33"], "cell_type": ["A", "B", "C"], "batch": ["1", "2", "3"]},
                index=["cell1", "cell2", "cell3"],
            ),
            pd.DataFrame(
                {
                    "UniProtID_new": ["P23456", "P34567", "P45678"],
                    "gene_name": ["gene1", "gene2", "gene3"],
                    "UniProtID": ["P12345", "P23456", "P34567"],
                },
                index=["G1", "G2", "G3"],
            ),
            1,
            0,
            1,
        ),
        # 4.: 111
        (
            pd.DataFrame(
                {"G1": [1, 2, 3], "G2": [4, 5, 6], "G3": [7, 8, 9]},
                index=["cell1", "cell2", "cell3"],
            ),
            pd.DataFrame(
                {"batch_new": ["11", "22", "33"], "cell_type": ["A", "B", "C"], "batch": ["1", "2", "3"]},
                index=["cell1", "cell2", "cell3"],
            ),
            pd.DataFrame(
                {
                    "UniProtID_new": ["P23456", "P34567", "P45678"],
                    "gene_name": ["gene1", "gene2", "gene3"],
                    "UniProtID": ["P12345", "P23456", "P34567"],
                },
                index=["G1", "G2", "G3"],
            ),
            1,
            1,
            1,
        ),
        # 5.: 000
        (
            pd.DataFrame(
                {"G1": [1, 3], "G3": [7, 9]},
                index=["cell1", "cell3"],
            ),
            pd.DataFrame(
                {"cell_type": ["A", "C"], "batch": ["1", "3"]},
                index=["cell1", "cell3"],
            ),
            pd.DataFrame(
                {
                    "gene_name": ["gene1", "gene3"],
                    "UniProtID": ["P12345", "P34567"],
                },
                index=["G1", "G3"],
            ),
            0,
            0,
            0,
        ),
        # 6.: 010
        (
            pd.DataFrame(
                {"G1": [1, 2, 3], "G2": [4, 5, 6], "G3": [7, 8, 9]},
                index=["cell1", "cell2", "cell3"],
            ),
            pd.DataFrame(
                {"cell_type": ["A", np.nan, "C"], "batch": ["1", np.nan, "3"]},
                index=["cell1", "cell2", "cell3"],
            ),
            pd.DataFrame(
                {
                    "gene_name": ["gene1", np.nan, "gene3"],
                    "UniProtID": ["P12345", np.nan, "P34567"],
                },
                index=["G1", "G2", "G3"],
            ),
            0,
            1,
            0,
        ),
        # 7.: 001
        (
            pd.DataFrame(
                {"G1": [1, 3], "G3": [7, 9]},
                index=["cell1", "cell3"],
            ),
            pd.DataFrame(
                {"batch_new": ["11", "33"], "cell_type": ["A", "C"], "batch": ["1", "3"]},
                index=["cell1", "cell3"],
            ),
            pd.DataFrame(
                {
                    "UniProtID_new": ["P23456", "P45678"],
                    "gene_name": ["gene1", "gene3"],
                    "UniProtID": ["P12345", "P34567"],
                },
                index=["G1", "G3"],
            ),
            0,
            0,
            1,
        ),
        # 8.: 011
        (
            pd.DataFrame(
                {"G1": [1, 2, 3], "G2": [4, 5, 6], "G3": [7, 8, 9]},
                index=["cell1", "cell2", "cell3"],
            ),
            pd.DataFrame(
                {"batch_new": ["11", "22", "33"], "cell_type": ["A", np.nan, "C"], "batch": ["1", np.nan, "3"]},
                index=["cell1", "cell2", "cell3"],
            ),
            pd.DataFrame(
                {
                    "UniProtID_new": ["P23456", "P34567", "P45678"],
                    "gene_name": ["gene1", np.nan, "gene3"],
                    "UniProtID": ["P12345", np.nan, "P34567"],
                },
                index=["G1", "G2", "G3"],
            ),
            0,
            1,
            1,
        ),
    ],
)
def test_add_metadata(
    example_data,
    example_sample_metadata,
    example_feature_metadata,
    expected_data,
    expected_sample_metadata,
    expected_feature_metadata,
    metadata_size,
    keep_data_shape,
    keep_existing_metadata,
):
    """"""

    # Restrict example metadata to the required size
    if metadata_size == 1:
        df = example_data.copy()
        sample_metadata = example_sample_metadata.copy()
        feature_metadata = example_feature_metadata.copy()
    elif metadata_size == 0:
        df = example_data.copy()
        sample_metadata = example_sample_metadata.loc[["cell1", "cell3"], :].copy()
        feature_metadata = example_feature_metadata.loc[["G1", "G3"], :].copy()

    # create original copies of the data and metadata to assert that the following operations do not change them
    df_original = df.copy()
    sample_metadata_original = sample_metadata.copy()
    feature_metadata_original = feature_metadata.copy()

    # create AnnData object (this would already be done during data loading; here substituted with a private method)
    adata = _to_anndata(df)

    # add some existing metadata to check whether it is kept
    adata.obs = pd.DataFrame({"batch_new": ["11", "22", "33"]}, index=["cell1", "cell2", "cell3"])
    adata.var = pd.DataFrame({"UniProtID_new": ["P23456", "P34567", "P45678"]}, index=["G1", "G2", "G3"])

    # Add metadata to data
    adata = apt.pp.add_metadata(
        adata, sample_metadata, axis=0, keep_data_shape=keep_data_shape, keep_existing_metadata=keep_existing_metadata
    )
    adata = apt.pp.add_metadata(
        adata, feature_metadata, axis=1, keep_data_shape=keep_data_shape, keep_existing_metadata=keep_existing_metadata
    )

    # main tests for data, sample-, and feature metadata
    assert adata.to_df().equals(expected_data), "Data should be aligned with sample and feature metadata"
    assert adata.obs.equals(expected_sample_metadata), "Sample metadata should be aligned with data"
    assert adata.var.equals(expected_feature_metadata), "Feature metadata should be aligned with data"

    # assert whether input data was changed
    assert df_original.equals(example_data), "Data should not be changed by adding it to Data object"
    assert sample_metadata.equals(sample_metadata_original), (
        "Sample metadata should not be changed by adding it to Data object"
    )
    assert feature_metadata.equals(feature_metadata_original), (
        "Feature metadata should not be changed by adding it to Data object"
    )


# Test proper failing behavior if resulting anndata object would be empty
@pytest.mark.parametrize(
    ("axis", "mismatching_metadata"),
    [
        (0, True),
        (1, True),
        (0, False),
        (1, False),
    ],
)
def test_add_metadata_nonmatching_sample_metadata(
    example_data,
    example_sample_metadata,
    example_feature_metadata,
    axis,
    mismatching_metadata,
):
    # get input datasets
    df = example_data.copy()

    # change sample metadata indices
    if axis == 0:
        md = example_sample_metadata.copy()
        if mismatching_metadata:
            md.index = md.index + "_changed"
    elif axis == 1:
        md = example_feature_metadata.copy()
        if mismatching_metadata:
            md.index = md.index + "_changed"

    # create AnnData object (this would already be done during data loading; here substituted with a private method)
    adata = _to_anndata(df)

    # If indices do not overlap, raise an error and do not change the incoming adata object
    if mismatching_metadata:
        adata_before = adata.copy()
        with pytest.raises(ValueError):
            # when
            adata = apt.pp.add_metadata(adata, md, axis=axis)
        assert adata.obs.equals(adata_before.obs)
        assert adata.var.equals(adata_before.var)
        assert np.array_equal(adata.X, adata_before.X)
    else:
        adata = apt.pp.add_metadata(adata, md, axis=axis)


# Test handling of incoming columns that overlap with existing metadata
@pytest.mark.parametrize(
    ("metadata", "inplace_metadata", "verbose", "expected_result", "expected_warning"),
    [
        # Test case 1: No overlapping columns
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            pd.DataFrame({"C": [5, 6], "D": [7, 8]}),
            True,
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            None,
        ),
        # Test case 2: Partial overlap
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            pd.DataFrame({"B": [5, 6], "C": [7, 8]}),
            True,
            pd.DataFrame({"A": [1, 2]}),
            "pp.add_metadata(): Synonymous fields, dropping ['B'] from incoming metadata.",
        ),
        # Test case 3: Complete overlap
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            pd.DataFrame({"A": [5, 6], "B": [7, 8]}),
            True,
            pd.DataFrame(index=[0, 1]),
            "pp.add_metadata(): Synonymous fields, dropping ['A', 'B'] from incoming metadata.",
        ),
        # Test case 4: Verbose is False, no warnings
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            pd.DataFrame({"B": [5, 6], "C": [7, 8]}),
            False,
            pd.DataFrame({"A": [1, 2]}),
            None,
        ),
    ],
)
def test_handle_overlapping_columns(metadata, inplace_metadata, verbose, expected_result, expected_warning):
    if expected_warning:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # when
            result = _handle_overlapping_columns(metadata, inplace_metadata, verbose=verbose)
            assert result.equals(expected_result)
            assert len(w) == 1
            assert expected_warning in str(w[0].message)
    else:
        with warnings.catch_warnings(record=True) as w:
            # when
            result = _handle_overlapping_columns(metadata, inplace_metadata, verbose=verbose)
            assert result.equals(expected_result)
            assert len(w) == 0


# test filtering of data based on metadata

# TODO: better test logic? Combining parameters exhaustively leads to a huge number or tests


@pytest.fixture
def adata_for_filtering():
    def make_dummy_data():
        size = 5
        # important: unique values for each row and column index
        df = pd.DataFrame(
            data=np.ones((size, size)),
            index=[f"cell{i + 1}" for i in range(size)],
            columns=[f"G{i + 1}" for i in range(size)],
        )
        sample_md = pd.DataFrame(
            {
                "sample_level": ["A", "A", "B", "B", "C"],
                "sample_level_na": ["A", None, "B", "B", "C"],
                "sample_value": [1, 2, 1, 2, 3],
                "sample_value_na": [1, 2, 1, np.nan, 3],
            },
            index=df.index,
        )
        feature_md = pd.DataFrame(
            {
                "feature_level": ["X", "X", "Y", "Y", "Z"],
                "feature_level_na": ["X", None, "Y", "Y", "Z"],
                "feature_value": [10, 20, 10, 20, 30],
                "feature_value_na": [10, 20, 10, np.nan, 30],
            },
            index=df.columns,
        )
        adata = apt.pp.data._to_anndata(df)
        adata = apt.pp.add_metadata(adata, sample_md, axis=0)
        return apt.pp.add_metadata(adata, feature_md, axis=1)

    return make_dummy_data()


# 1. Establish that filtering works with "keep" and "drop" settings on sample and feature metadata
@pytest.mark.parametrize(
    ("expected_adata_index", "filter_dict", "axis", "logic", "action"),
    [
        # 1.1. Sample metadata establish basic functionality with "keep" setting
        # 1.1.1. "and" works on strings & numbers, strings & lists and strings & tuples
        (
            np.array(["cell1"]),
            {"sample_level": "A", "sample_value": 1},
            0,
            "and",
            "keep",
        ),
        (
            np.array(["cell1", "cell2"]),
            {"sample_level": "A", "sample_value": [1, 2]},
            0,
            "and",
            "keep",
        ),
        (
            np.array(["cell1", "cell2"]),
            {"sample_level": "A", "sample_value": (1, 3)},
            0,
            "and",
            "keep",
        ),
        # 1.1.2. "or" works on strings & numbers, strings & lists and strings & tuples
        (
            np.array(["cell1", "cell2", "cell3"]),
            {"sample_level": "A", "sample_value": 1},
            0,
            "or",
            "keep",
        ),
        (
            np.array(["cell1", "cell2", "cell3", "cell4"]),
            {"sample_level": "A", "sample_value": [1, 2]},
            0,
            "or",
            "keep",
        ),
        (
            np.array(["cell1", "cell2", "cell3", "cell4"]),
            {"sample_level": "A", "sample_value": (1, 3)},
            0,
            "or",
            "keep",
        ),
        # 1.2. establish basic functionality with "drop" setting
        # 1.2.1. "and" works on strings & numbers, strings & lists and strings & tuples
        (
            np.array(["cell2", "cell3", "cell4", "cell5"]),
            {"sample_level": "A", "sample_value": 1},
            0,
            "and",
            "drop",
        ),
        (
            np.array(["cell3", "cell4", "cell5"]),
            {"sample_level": "A", "sample_value": [1, 2]},
            0,
            "and",
            "drop",
        ),
        (
            np.array(["cell3", "cell4", "cell5"]),
            {"sample_level": "A", "sample_value": (1, 3)},
            0,
            "and",
            "drop",
        ),
        # 1.2.2. "or" works on strings & numbers, strings & lists and strings & tuples
        (
            np.array(["cell4", "cell5"]),
            {"sample_level": "A", "sample_value": 1},
            0,
            "or",
            "drop",
        ),
        (
            np.array(["cell5"]),
            {"sample_level": "A", "sample_value": [1, 2]},
            0,
            "or",
            "drop",
        ),
        (
            np.array(["cell5"]),
            {"sample_level": "A", "sample_value": (1, 3)},
            0,
            "or",
            "drop",
        ),
        # 2.1. Feature metadata: establish basic functionality with "keep" setting
        # 2.1.1. "and" works on strings & numbers, strings & lists and strings & tuples
        (
            np.array(["G1"]),
            {"feature_level": "X", "feature_value": 10},
            1,
            "and",
            "keep",
        ),
        (
            np.array(["G1", "G2"]),
            {"feature_level": "X", "feature_value": [10, 20]},
            1,
            "and",
            "keep",
        ),
        (
            np.array(["G1", "G2"]),
            {"feature_level": "X", "feature_value": (10, 30)},
            1,
            "and",
            "keep",
        ),
        # 2.1.2. "or" works on strings & numbers, strings & lists and strings & tuples
        (
            np.array(["G1", "G2", "G3"]),
            {"feature_level": "X", "feature_value": 10},
            1,
            "or",
            "keep",
        ),
        (
            np.array(["G1", "G2", "G3", "G4"]),
            {"feature_level": "X", "feature_value": [10, 20]},
            1,
            "or",
            "keep",
        ),
        (
            np.array(["G1", "G2", "G3", "G4"]),
            {"feature_level": "X", "feature_value": (10, 30)},
            1,
            "or",
            "keep",
        ),
        # 2.2. establish basic functionality with "drop" setting
        # 2.2.1. "and" works strings & numbers, strings & lists and strings & tuples
        (
            np.array(["G2", "G3", "G4", "G5"]),
            {"feature_level": "X", "feature_value": 10},
            1,
            "and",
            "drop",
        ),
        (
            np.array(["G3", "G4", "G5"]),
            {"feature_level": "X", "feature_value": [10, 20]},
            1,
            "and",
            "drop",
        ),
        (
            np.array(["G3", "G4", "G5"]),
            {"feature_level": "X", "feature_value": (10, 30)},
            1,
            "and",
            "drop",
        ),
        # 2.2.2. "or" works on strings & numbers, strings & lists and strings & tuples
        (
            np.array(["G4", "G5"]),
            {"feature_level": "X", "feature_value": 10},
            1,
            "or",
            "drop",
        ),
        (
            np.array(["G5"]),
            {"feature_level": "X", "feature_value": [10, 20]},
            1,
            "or",
            "drop",
        ),
        (
            np.array(["G5"]),
            {"feature_level": "X", "feature_value": (10, 30)},
            1,
            "or",
            "drop",
        ),
        # 3. Evaluate special cases on sample metadata
        # 3.1. tuple range open to the right
        (
            np.array(["cell2", "cell4", "cell5"]),
            {"sample_value": (2, None)},
            0,
            "and",
            "keep",
        ),
        # 3.2. tuple range open to the left
        (
            np.array(["cell1", "cell2", "cell3", "cell4"]),
            {"sample_value": (None, 3)},
            0,
            "and",
            "keep",
        ),
        # 3.3. tuple range open on both sides
        (
            np.array(["cell1", "cell2", "cell3", "cell4", "cell5"]),
            {"sample_value": (None, None)},
            0,
            "and",
            "keep",
        ),
        # 3.4. empty filter_dict
        (
            np.array(["cell1", "cell2", "cell3", "cell4", "cell5"]),
            {},
            0,
            "and",
            "keep",
        ),
        # 3.5. no matches to keep
        (
            np.array([]),
            {"sample_level": "E", "sample_value": 99},
            0,
            "and",
            "keep",
        ),
        # 3.6. no matches to drop
        (
            np.array(["cell1", "cell2", "cell3", "cell4", "cell5"]),
            {"sample_level": "E", "sample_value": 99},
            0,
            "and",
            "drop",
        ),
        # 3.7. all data removed
        (
            np.array([]),
            {"sample_level": ["A", "B", "C"]},
            0,
            "or",
            "drop",
        ),
        # 3.8. NA in numeric column (tuple based filtering)
        (
            np.array(["cell1", "cell2", "cell3", "cell5"]),
            {"sample_value_na": (1, 4)},
            0,
            "and",
            "keep",
        ),
        # 3.9. NA in numeric column (list based filtering)
        (
            np.array(["cell1", "cell2", "cell3"]),
            {"sample_value_na": [1, 2]},
            0,
            "and",
            "keep",
        ),
        # 3.10. NA in string column (string based filtering)
        (
            np.array(["cell1"]),
            {"sample_level_na": "A"},
            0,
            "and",
            "keep",
        ),
        # 3.11. NA in string column (list based filtering)
        (
            np.array(["cell1", "cell3", "cell4"]),
            {"sample_level_na": ["A", "B"]},
            0,
            "and",
            "keep",
        ),
    ],
)
def test_filter_by_metadata(adata_for_filtering, expected_adata_index, filter_dict, axis, logic, action):
    adata = adata_for_filtering.copy()
    # when
    adata = apt.pp.filter_by_metadata(adata, filter_dict, axis=axis, logic=logic, action=action)
    # then
    if len(expected_adata_index) == 0:
        assert adata.n_obs == 0 if axis == 0 else adata.n_vars == 0
    else:
        assert np.array_equal(adata.obs.index if axis == 0 else adata.var.index, expected_adata_index)


class TestScaleAndCenter:
    @pytest.fixture
    def anndata_scale_and_center(self, example_data) -> tuple[ad.AnnData, dict[str, pd.DataFrame]]:
        """Generate example anndata with ground truths"""
        adata = _to_anndata(example_data)
        adata.layers["new_layer"] = adata.X.copy()

        expected = {
            "standard": pd.DataFrame(
                {"G1": [-1.224745, 0.0, 1.224745], "G2": [-1.224745, 0.0, 1.224745], "G3": [-1.224745, 0.0, 1.224745]},
                index=["cell1", "cell2", "cell3"],
            ),
            "robust": pd.DataFrame(
                {"G1": [-1.0, 0.0, 1.0], "G2": [-1.0, 0.0, 1.0], "G3": [-1.0, 0.0, 1.0]},
                index=["cell1", "cell2", "cell3"],
            ),
        }

        return adata, expected

    @pytest.mark.parametrize("layer", [None, "new_layer"])
    @pytest.mark.parametrize("scaler", ["standard", "robust"])
    def test_scale_and_center_inplace(self, anndata_scale_and_center, scaler: str, layer: str) -> None:
        """Test that alphapepttools.pp.scale_and_center modifies anndata correctly inplace"""
        adata, expected = anndata_scale_and_center

        return_value = apt.pp.scale_and_center(adata, scaler=scaler, layer=layer, copy=False)

        assert return_value is None  # inplace expected

        if layer is None:
            assert np.all(np.isclose(adata.X, expected[scaler].values))
        else:
            assert np.all(np.isclose(adata.layers[layer], expected[scaler].values))

    @pytest.mark.parametrize("layer", [None, "new_layer"])
    @pytest.mark.parametrize("scaler", ["standard", "robust"])
    def test_scale_and_center_copy(self, anndata_scale_and_center, scaler: str, layer: str) -> None:
        """Test that alphapepttools.pp.scale_and_center correctly returns a copy"""
        adata, expected = anndata_scale_and_center
        adata_original = adata.copy()

        adata_new = apt.pp.scale_and_center(adata, scaler=scaler, layer=layer, copy=True)

        assert isinstance(adata_new, ad.AnnData)

        if layer is None:
            assert np.all(np.isclose(adata_new.X, expected[scaler].values))
            # Original object was not modified
            assert np.all(np.isclose(adata.X, adata_original.X))

        else:
            assert np.all(np.isclose(adata_new.layers[layer], expected[scaler].values))
            # Original object was not modified
            assert np.all(np.isclose(adata.layers[layer], adata_original.layers[layer]))


@pytest.fixture
def data_test_completeness_filter():
    def make_dummy_data():
        X = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [np.nan, 7, 6, 7, 8],
                "C": [np.nan, np.nan, 9, 10, 11],
                "D": [np.nan, np.nan, np.nan, 13, 14],
                "E": [np.nan, np.nan, np.nan, np.nan, 17],
            },
            index=["cell1", "cell2", "cell3", "cell4", "cell5"],
        )
        sample_metadata = pd.DataFrame(
            {
                "batch": ["1", "1", "1", "2", "2"],
            },
            index=["cell1", "cell2", "cell3", "cell4", "cell5"],
        )
        feature_metadata = pd.DataFrame(
            {
                "gene_name": ["GO1", "GO1", "GO1", "GO2", "GO2"],
            },
            index=["A", "B", "C", "D", "E"],
        )
        adata = _to_anndata(X)
        adata.obs = sample_metadata
        adata.var = feature_metadata
        return adata

    return make_dummy_data()


# test data completeness filtering
@pytest.mark.parametrize(
    ("expected_columns", "expected_rows", "max_missing_kwargs", "group_column", "groups", "action", "keep_strategy"),
    [
        # 1. Check filtering of columns (features)
        # 1.1. Filter columns with 0.5 threshold
        (
            ["A", "B", "C"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.5},
            None,
            None,
            "drop",
            "all",
        ),
        # 1.2. Filter columns with 0.6 threshold so that one value lies exactly on the threshold --> this should be kept since ">" is used
        (
            ["A", "B", "C", "D"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.6},
            None,
            None,
            "drop",
            "all",
        ),
        # 1.3. flag the columns with 0.5 threshold - not to drop them
        (
            ["A", "B", "C", "D", "E"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.5},
            None,
            None,
            "flag",
            "all",
        ),
        # 1.4. Filter columns with 1.0 threshold: keep all columns
        (
            ["A", "B", "C", "D", "E"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 1.0},
            None,
            None,
            "drop",
            "all",
        ),
        # 1.5. Filter columns with 0.0 threshold: remove columns with any missing values
        (
            ["A"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.0},
            None,
            None,
            "drop",
            "all",
        ),
        # 2. Group-wise filtering
        # 2.1. Group by 'batch' and filter columns with 0.5 threshold
        (
            ["A", "B"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.5},
            "batch",
            None,
            "drop",
            "all",
        ),
        # 2.2. Group by 'batch' and filter columns with 1.0 threshold: keep all columns
        (
            ["A", "B", "C", "D", "E"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 1.0},
            "batch",
            None,
            "drop",
            "all",
        ),
        # 2.3. Group by 'batch' and filter columns with 0.0 threshold: remove columns with any missing values in either batch
        (
            ["A"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.0},
            "batch",
            None,
            "drop",
            "all",
        ),
        # 3. Group-wise filtering with specific groups
        # 3.1. Group by 'batch' and filter only batch '2' with 0.5 threshold
        (
            ["A", "B", "C", "D", "E"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.5},
            "batch",
            ["2"],
            "drop",
            "all",
        ),
        # 3.2. Group by 'batch' and filter only batch '2' with 1.0 threshold: keep all columns
        (
            ["A", "B", "C", "D", "E"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 1.0},
            "batch",
            ["2"],
            "drop",
            "all",
        ),
        # 3.3. Group by 'batch' and filter only batch '2' with 0.0 threshold: remove columns with any missing values in that group
        (
            ["A", "B", "C", "D"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.0},
            "batch",
            ["2"],
            "drop",
            "all",
        ),
        # 3.4. Group by 'batch' and filter only batch '1' with 0.5 threshold
        (
            ["A", "B"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.5},
            "batch",
            ["1"],
            "drop",
            "all",
        ),
        # 3.5. Group by 'batch' and filter only batch '1' with 1.0 threshold: keep all columns
        (
            ["A", "B", "C", "D", "E"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 1.0},
            "batch",
            ["1"],
            "drop",
            "all",
        ),
        # 3.6. Group by 'batch' and filter only batch '1' with 0.0 threshold: remove columns with any missing values in that group
        (
            ["A"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.0},
            "batch",
            ["1"],
            "drop",
            "all",
        ),
        # 4. Test with two groups specified (should be the same as when only the 'batch' column is specified)
        # 4.1. Group by 'batch' and filter batches '1' and '2' with 0.5 threshold
        (
            ["A", "B"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.5},
            "batch",
            ["1", "2"],
            "drop",
            "all",
        ),
        # 4.2. Group by 'batch' and filter batches '1' and '2' with 1.0 threshold: keep all columns
        (
            ["A", "B", "C", "D", "E"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 1.0},
            "batch",
            ["1", "2"],
            "drop",
            "all",
        ),
        # 4.3. Group by 'batch' and filter batches '1' and '2' with 0.0 threshold: remove columns with any missing values in that group
        (
            ["A"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.0},
            "batch",
            ["1", "2"],
            "drop",
            "all",
        ),
        # 5. Group-wise filtering with keep_strategy="any" (logical OR) — diverges from "all"
        # 5.1. max_missing=0.5: batch 2 passes everything (<=0.5), so "any" keeps all features
        # (vs. "all" which gives ["A", "B"] — see 2.1)
        (
            ["A", "B", "C", "D", "E"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.5},
            "batch",
            None,
            "drop",
            "any",
        ),
        # 5.2. max_missing=0.0 with "any": A passes in batch 1; A,B,C,D pass in batch 2
        # → keep A,B,C,D (vs. "all" which gives ["A"] — see 2.3)
        (
            ["A", "B", "C", "D"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.0},
            "batch",
            None,
            "drop",
            "any",
        ),
        # 5.3. Single selected group: "any" reduces to the same result as "all"
        (
            ["A", "B"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.5},
            "batch",
            ["1"],
            "drop",
            "any",
        ),
        # 5.4. Two groups explicit: "any" matches the groups=None case 5.1
        (
            ["A", "B", "C", "D", "E"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_fraction": 0.5},
            "batch",
            ["1", "2"],
            "drop",
            "any",
        ),
        # 6. max_missing_count: absolute count of missing values allowed.
        # Per-feature missing counts (out of 5): A=0, B=1, C=2, D=3, E=4
        # 6.1. max_missing_count=0: keep only fully-complete features
        (
            ["A"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_count": 0},
            None,
            None,
            "drop",
            "all",
        ),
        # 6.2. max_missing_count=1: allow at most 1 missing value
        #      (contrast with 1.4: max_missing_fraction=1.0 keeps everything)
        (
            ["A", "B"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_count": 1},
            None,
            None,
            "drop",
            "all",
        ),
        # 6.3. max_missing_count=3: allow at most 3 missing values
        (
            ["A", "B", "C", "D"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_count": 3},
            None,
            None,
            "drop",
            "all",
        ),
        # 6.4. Group-wise count, keep_strategy="all": pass if <=1 missing in *every* batch.
        #      Batch 1 (3 samples) missing: A=0,B=1,C=2,D=3,E=3 ; Batch 2 (2 samples): A=0,B=0,C=0,D=0,E=1
        #      → only A,B satisfy <=1 in both batches
        (
            ["A", "B"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_count": 1},
            "batch",
            None,
            "drop",
            "all",
        ),
        # 6.5. Group-wise count, keep_strategy="any": pass if <=1 missing in *at least one* batch.
        #      Batch 2 has <=1 missing for all features → keep everything
        (
            ["A", "B", "C", "D", "E"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            {"max_missing_count": 1},
            "batch",
            None,
            "drop",
            "any",
        ),
    ],
)
def test_filter_data_completeness(
    data_test_completeness_filter,
    expected_columns,
    expected_rows,
    max_missing_kwargs,
    group_column,
    groups,
    action,
    keep_strategy,
):
    # given
    adata = data_test_completeness_filter.copy()

    # when
    adata_filtered = apt.pp.filter_data_completeness(
        adata=adata,
        **max_missing_kwargs,
        group_column=group_column,
        groups=groups,
        action=action,
        keep_strategy=keep_strategy,
    )

    # then
    assert adata_filtered.var.index.to_list() == expected_columns
    assert adata_filtered.obs.index.to_list() == expected_rows

    # assert whether input data was changed
    assert adata.var.index.to_list() == data_test_completeness_filter.var.index.to_list()
    assert adata.obs.index.to_list() == data_test_completeness_filter.obs.index.to_list()
    assert np.array_equal(adata.X, data_test_completeness_filter.X, equal_nan=True)


def test_filter_data_completeness_invalid_keep_strategy(data_test_completeness_filter):
    with pytest.raises(ValueError, match="Supported keep_strategies"):
        apt.pp.filter_data_completeness(
            adata=data_test_completeness_filter,
            max_missing_fraction=0.5,
            group_column="batch",
            keep_strategy="invalid",
        )


def test_filter_data_completeness_fraction_out_of_range(data_test_completeness_filter):
    with pytest.raises(ValueError, match="between 0 and 1"):
        apt.pp.filter_data_completeness(data_test_completeness_filter, max_missing_fraction=1.5)


def test_filter_data_completeness_negative_count(data_test_completeness_filter):
    with pytest.raises(ValueError, match="non-negative"):
        apt.pp.filter_data_completeness(data_test_completeness_filter, max_missing_count=-1)


def test_filter_data_completeness_both_thresholds(data_test_completeness_filter):
    # the two thresholds are contradictory, so passing both is an error rather than a silent preference
    with pytest.raises(ValueError, match="Exactly one of"):
        apt.pp.filter_data_completeness(data_test_completeness_filter, max_missing_fraction=0.5, max_missing_count=1)


def test_filter_data_completeness_no_threshold(data_test_completeness_filter):
    with pytest.raises(ValueError, match="Exactly one of"):
        apt.pp.filter_data_completeness(data_test_completeness_filter)


def test_filter_data_completeness_removed_max_missing(data_test_completeness_filter):
    # the removed argument gets a migration hint rather than the "exactly one threshold" error
    with pytest.raises(TypeError, match="`max_missing` has been replaced"):
        apt.pp.filter_data_completeness(data_test_completeness_filter, max_missing=0.5, action="drop")


def test_filter_data_completeness_unknown_kwarg(data_test_completeness_filter):
    # **kwargs must not silently swallow typos
    with pytest.raises(TypeError, match="Unexpected keyword argument"):
        apt.pp.filter_data_completeness(data_test_completeness_filter, max_missing_fraction=0.5, actoin="drop")


# the mode follows the argument that was passed, never the runtime type of its value
@pytest.mark.parametrize(
    ("expected_columns", "max_missing_kwargs"),
    [
        # an int fraction is still a fraction: 1 means 100% missing allowed -> keep everything
        (["A", "B", "C", "D", "E"], {"max_missing_fraction": 1}),
        # a numpy integer count behaves exactly like the equivalent python int (see case 6.2)
        (["A", "B"], {"max_missing_count": np.int64(1)}),
        # a numpy float fraction behaves exactly like the equivalent python float (see case 1.1)
        (["A", "B", "C"], {"max_missing_fraction": np.float32(0.5)}),
    ],
)
def test_filter_data_completeness_threshold_type_does_not_select_mode(
    data_test_completeness_filter, expected_columns, max_missing_kwargs
):
    adata_filtered = apt.pp.filter_data_completeness(data_test_completeness_filter, **max_missing_kwargs, action="drop")
    assert adata_filtered.var.index.to_list() == expected_columns


def test_filter_data_completeness_unused_categories_ignored(data_test_completeness_filter):
    # unused levels of a categorical group column must not contribute empty groups:
    # an empty group has no missing values to count, and its mean is NaN
    adata = data_test_completeness_filter.copy()
    adata.obs["batch"] = pd.Categorical(adata.obs["batch"], categories=["1", "2", "3"])

    for max_missing_kwargs in ({"max_missing_fraction": 0.5}, {"max_missing_count": 1}):
        for keep_strategy in ("all", "any"):
            with_unused = apt.pp.filter_data_completeness(
                adata, **max_missing_kwargs, group_column="batch", keep_strategy=keep_strategy, action="drop"
            )
            without_unused = apt.pp.filter_data_completeness(
                data_test_completeness_filter,
                **max_missing_kwargs,
                group_column="batch",
                keep_strategy=keep_strategy,
                action="drop",
            )
            assert with_unused.var.index.to_list() == without_unused.var.index.to_list()


# test _resolve_max_missing
@pytest.mark.parametrize(
    ("max_missing_fraction", "max_missing_count", "expected"),
    [
        (0.0, None, (0.0, False)),
        (0.5, None, (0.5, False)),
        (1.0, None, (1.0, False)),
        (None, 0, (0, True)),
        (None, 1, (1, True)),
        (None, 5, (5, True)),
        # an int fraction stays a fraction, a numpy scalar is normalised to a python number
        (1, None, (1.0, False)),
        (None, np.int64(2), (2, True)),
    ],
)
def test_resolve_max_missing(max_missing_fraction, max_missing_count, expected):
    threshold, is_count_mode = _resolve_max_missing(max_missing_fraction, max_missing_count)

    assert (threshold, is_count_mode) == expected
    assert isinstance(threshold, int if is_count_mode else float)


@pytest.mark.parametrize(
    ("max_missing_fraction", "max_missing_count", "match"),
    [
        (1.5, None, "between 0 and 1"),
        (-0.1, None, "between 0 and 1"),
        (None, -1, "non-negative"),
        (0.5, 1, "Exactly one of"),
        (None, None, "Exactly one of"),
    ],
)
def test_resolve_max_missing_invalid(max_missing_fraction, max_missing_count, match):
    with pytest.raises(ValueError, match=match):
        _resolve_max_missing(max_missing_fraction, max_missing_count)


# test _count_or_fraction_missing
def test_count_or_fraction_missing():
    # per-feature missing values: A=0, B=1, C=2 (out of 3 rows)
    x = np.array([[1.0, np.nan, np.nan], [2.0, 5.0, np.nan], [3.0, 6.0, 9.0]])

    # is_count_mode=True -> absolute counts
    np.testing.assert_array_equal(_count_or_fraction_missing(x, is_count_mode=True), [0, 1, 2])

    # is_count_mode=False -> fractions
    np.testing.assert_allclose(_count_or_fraction_missing(x, is_count_mode=False), [0.0, 1 / 3, 2 / 3])


# test data_column_to_array
@pytest.mark.parametrize(
    ("expected_array", "column", "transpose"),
    [
        # 1. Extracting columns from either anndata values or observation metadata
        # 1.1. Column is in var_names
        (
            np.array([1, 2, 3]),
            "G1",
            False,
        ),
        # 1.2. Column is in obs.columns
        (
            np.array(["1", "2", "3"]),
            "batch",
            False,
        ),
        # 2. Transposed adata, as if to access rows
        # 2.1. Column is in original obs_names
        (
            np.array([2, 5, 8]),
            "cell2",
            True,
        ),
        # 2.2. Column is in original var.columns
        (
            np.array(["gene1", "gene2", "gene3"]),
            "gene_name",
            True,
        ),
    ],
)
def test_data_column_to_array(
    example_data,
    example_sample_metadata,
    example_feature_metadata,
    expected_array,
    column,
    transpose,
):
    # given
    adata = _to_anndata(example_data)
    adata = apt.pp.add_metadata(adata, example_sample_metadata, axis=0)
    adata = apt.pp.add_metadata(adata, example_feature_metadata, axis=1)

    # when
    array = data_column_to_array(adata, column) if not transpose else data_column_to_array(adata.transpose(), column)

    # then
    assert np.all(array == expected_array)


### Test coerce_to_dataframe ###


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}, index=["row1", "row2", "row3"])


@pytest.fixture
def sample_anndata(sample_dataframe):
    """Create a sample AnnData object for testing"""
    # Create AnnData with X matrix and some obs data
    return ad.AnnData(
        X=sample_dataframe.values,
        obs=pd.DataFrame({"cell_type": ["type_A", "type_B", "type_C"]}, index=sample_dataframe.index),
        var=pd.DataFrame({"gene_name": sample_dataframe.columns.tolist()}, index=sample_dataframe.columns),
    )


@pytest.mark.parametrize(
    ("input_type", "expected_shape", "expected_columns"),
    [
        ("dataframe", (3, 3), ["A", "B", "C"]),  # DataFrame input
        ("anndata", (3, 4), ["A", "B", "C", "cell_type"]),  # AnnData input includes obs columns
    ],
)
def test_coerce_to_dataframe(
    sample_dataframe,
    sample_anndata,
    input_type,
    expected_shape,
    expected_columns,
):
    """Test that coerce_to_dataframe correctly handles DataFrame and AnnData inputs"""
    # given
    data = sample_dataframe if input_type == "dataframe" else sample_anndata

    # when
    result = coerce_to_dataframe(data)

    # then
    assert isinstance(result, pd.DataFrame)
    assert result.shape == expected_shape
    assert list(result.columns) == expected_columns

    # Check index is preserved
    assert list(result.index) == ["row1", "row2", "row3"]

    # Check data values for the main columns
    if input_type == "dataframe":
        pd.testing.assert_frame_equal(result, sample_dataframe)
    else:  # anndata
        # Check that X values are preserved
        pd.testing.assert_frame_equal(
            result[["A", "B", "C"]],
            pd.DataFrame(sample_anndata.X, index=sample_anndata.obs.index, columns=["A", "B", "C"]),
        )
        # Check that obs columns are included
        assert list(result["cell_type"]) == ["type_A", "type_B", "type_C"]
