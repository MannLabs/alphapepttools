import warnings

import numpy as np
import pandas as pd
import pytest

import alphatools as at

# import private method to obtain anndata object
from alphatools.pp.data import _get_df_from_adata, _handle_overlapping_columns, _to_anndata

### Fixtures ###


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
        adata = at.pp.data._to_anndata(example_data())
        at.pp.add_metadata(adata, example_sample_metadata(), axis=0)
        at.pp.add_metadata(adata, example_feature_metadata(), axis=1)
        return adata

    return make_dummy_data()


### Tests ###


# test adding metadata to data
@pytest.mark.parametrize(
    (
        "expected_data",
        "expected_sample_metadata",
        "expected_feature_metadata",
        "keep_data_shape",
        "keep_existing_metadata",
    ),
    [
        # add sample and feature metadata to data and keep only data shape
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
            False,
            False,
        ),
        # Add metadata with left join to data, which forces metadata to be padded with NAs
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
            True,
            False,
        ),
        # add metadata with inner join to already existing metadata
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
            False,
            True,
        ),
        # add metadata with left join to already existing metadata
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
            True,
            True,
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
    keep_data_shape,
    keep_existing_metadata,
):
    """"""
    # get input datasets
    if not keep_data_shape:
        df = example_data.copy()
        sample_metadata = example_sample_metadata.copy()
        feature_metadata = example_feature_metadata.copy()
    else:
        df = example_data.copy()
        sample_metadata = example_sample_metadata.copy().iloc[:2, :]  # keep samples cell1 and cell3
        feature_metadata = example_feature_metadata.copy().iloc[:2, :]  # keep features G1 and G3

    # create AnnData object (this would already be done during data loading; here substituted with a private method)
    adata = _to_anndata(df)

    # add some existing metadata to check whether it is kept
    adata.obs = pd.DataFrame({"batch_new": ["11", "22", "33"]}, index=["cell1", "cell2", "cell3"])
    adata.var = pd.DataFrame({"UniProtID_new": ["P23456", "P34567", "P45678"]}, index=["G1", "G2", "G3"])

    # Add metadata to data
    # when
    adata = at.pp.add_metadata(
        adata, sample_metadata, axis=0, keep_data_shape=keep_data_shape, keep_existing_metadata=keep_existing_metadata
    )
    adata = at.pp.add_metadata(
        adata, feature_metadata, axis=1, keep_data_shape=keep_data_shape, keep_existing_metadata=keep_existing_metadata
    )

    # check whether data was correctly added and aligned
    df_aligned = _get_df_from_adata(adata)
    sample_metadata_aligned = adata.obs
    feature_metadata_aligned = adata.var

    # main tests for data, sample-, and feature metadata
    assert df_aligned.equals(expected_data), "Data should be aligned with sample and feature metadata"
    assert sample_metadata_aligned.equals(expected_sample_metadata), "Sample metadata should be aligned with data"
    assert feature_metadata_aligned.equals(expected_feature_metadata), "Feature metadata should be aligned with data"

    # assert whether input data was changed
    assert df.equals(example_data), "Data should not be changed by adding it to Data object"
    assert sample_metadata.equals(
        example_sample_metadata if not keep_data_shape else example_sample_metadata.iloc[:2, :]
    ), "Sample metadata should not be changed by adding it to Data object"
    assert feature_metadata.equals(
        example_feature_metadata if not keep_data_shape else example_feature_metadata.iloc[:2, :]
    ), "Feature metadata should not be changed by adding it to Data object"


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
            adata = at.pp.add_metadata(adata, md, axis=axis)
        assert adata.obs.equals(adata_before.obs)
        assert adata.var.equals(adata_before.var)
        assert np.array_equal(adata.X, adata_before.X)
    else:
        adata = at.pp.add_metadata(adata, md, axis=axis)


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
        adata = at.pp.data._to_anndata(df)
        adata = at.pp.add_metadata(adata, sample_md, axis=0)
        return at.pp.add_metadata(adata, feature_md, axis=1)

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
    adata = at.pp.filter_by_metadata(adata, filter_dict, axis=axis, logic=logic, action=action)
    # then
    if len(expected_adata_index) == 0:
        assert adata.n_obs == 0 if axis == 0 else adata.n_vars == 0
    else:
        assert np.array_equal(adata.obs.index if axis == 0 else adata.var.index, expected_adata_index)


# test scaling of data
@pytest.mark.parametrize(
    ("expected_data", "scaler", "from_layer", "to_layer"),
    [
        # standard scaler from X to X
        (
            pd.DataFrame(
                {"G1": [-1.224745, 0.0, 1.224745], "G2": [-1.224745, 0.0, 1.224745], "G3": [-1.224745, 0.0, 1.224745]},
                index=["cell1", "cell2", "cell3"],
            ),
            "standard",
            None,
            None,
        ),
        # robust from X to X
        (
            pd.DataFrame(
                {"G1": [-1.0, 0.0, 1.0], "G2": [-1.0, 0.0, 1.0], "G3": [-1.0, 0.0, 1.0]},
                index=["cell1", "cell2", "cell3"],
            ),
            "robust",
            None,
            None,
        ),
        # standard scaler from layer to X
        (
            pd.DataFrame(
                {"G1": [-1.224745, 0.0, 1.224745], "G2": [-1.224745, 0.0, 1.224745], "G3": [-1.224745, 0.0, 1.224745]},
                index=["cell1", "cell2", "cell3"],
            ),
            "standard",
            "source_layer",
            None,
        ),
        # standard scaler from X to layer
        (
            pd.DataFrame(
                {"G1": [-1.224745, 0.0, 1.224745], "G2": [-1.224745, 0.0, 1.224745], "G3": [-1.224745, 0.0, 1.224745]},
                index=["cell1", "cell2", "cell3"],
            ),
            "standard",
            None,
            "target_layer",
        ),
        # standard scaler from layer to layer
        (
            pd.DataFrame(
                {"G1": [-1.224745, 0.0, 1.224745], "G2": [-1.224745, 0.0, 1.224745], "G3": [-1.224745, 0.0, 1.224745]},
                index=["cell1", "cell2", "cell3"],
            ),
            "standard",
            "source_layer",
            "target_layer",
        ),
    ],
)
def test_pp_scale_and_center(
    example_data,
    expected_data,
    scaler,
    from_layer,
    to_layer,
):
    # get input datasets
    df = example_data.copy()

    # create AnnData object (this would already be done during data loading; here substituted with a private method)
    if from_layer is None:
        adata = _to_anndata(df)
    elif from_layer is not None:
        adata = _to_anndata(df)
        adata.layers[from_layer] = adata.X.copy()
        # remove X to test whether data is scaled correctly
        adata.X = 0

    # scale data
    at.pp.scale_and_center(adata, scaler=scaler, to_layer=to_layer, from_layer=from_layer)

    # check whether data was correctly scaled
    if to_layer is None:
        assert np.all(np.isclose(adata.X, expected_data.values))
    elif to_layer is not None:
        assert np.all(np.isclose(adata.layers[to_layer], expected_data.values))

    # assert whether input data was changed
    assert df.equals(example_data), "Data should not be changed by scaling"
