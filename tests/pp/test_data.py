import warnings

import numpy as np
import pandas as pd
import pytest

import alphatools as at

# import private method to obtain anndata object
from alphatools.pp.data import _adata_column_to_array, _handle_overlapping_columns, _to_anndata

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
    df_aligned = adata.to_df()
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
def test_scale_and_center(
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
    ("expected_columns", "expected_rows", "max_missing", "axis"),
    [
        # 1. Check filtering of columns (features)
        # 1.1. Filter columns with 0.5 threshold
        (
            ["A", "B", "C"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            0.5,
            1,
        ),
        # 1.2. Filter columns with 0.6 threshold so that one value lies exactly on the threshold --> this should be kept since ">" is used
        (
            ["A", "B", "C", "D"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            0.6,
            1,
        ),
        # 1.3. Filter columns with 1.0 threshold: keep all columns
        (
            ["A", "B", "C", "D", "E"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            1.0,
            1,
        ),
        # 1.4. Filter columns with 0.0 threshold: remove columns with any missing values
        (
            ["A"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            0.0,
            1,
        ),
        # 2. Check filtering of rows (samples)
        # 2.1. Filter rows with 0.5 threshold
        (
            ["A", "B", "C", "D", "E"],
            ["cell3", "cell4", "cell5"],
            0.5,
            0,
        ),
        # 2.2. Filter rows with 0.6 threshold so that one value lies exactly on the threshold --> this should be kept since ">" is used
        (
            ["A", "B", "C", "D", "E"],
            ["cell2", "cell3", "cell4", "cell5"],
            0.6,
            0,
        ),
        # 2.3. Filter rows with 1.0 threshold: keep all rows
        (
            ["A", "B", "C", "D", "E"],
            ["cell1", "cell2", "cell3", "cell4", "cell5"],
            1.0,
            0,
        ),
        # 2.4. Filter rows with 0.0 threshold: remove rows with any missing values
        (
            ["A", "B", "C", "D", "E"],
            ["cell5"],
            0.0,
            0,
        ),
    ],
)
def test_filter_data_completeness(
    data_test_completeness_filter,
    expected_columns,
    expected_rows,
    max_missing,
    axis,
):
    # given
    adata = data_test_completeness_filter.copy()

    # when
    adata_filtered = at.pp.filter_data_completeness(
        adata=adata,
        max_missing=max_missing,
        axis=axis,
    )

    # then
    assert adata_filtered.var.index.to_list() == expected_columns
    assert adata_filtered.obs.index.to_list() == expected_rows

    # assert whether input data was changed
    assert adata.var.index.to_list() == data_test_completeness_filter.var.index.to_list()
    assert adata.obs.index.to_list() == data_test_completeness_filter.obs.index.to_list()
    assert np.array_equal(adata.X, data_test_completeness_filter.X, equal_nan=True)


# test adata_column_to_array
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
def test_adata_column_to_array(
    example_data,
    example_sample_metadata,
    example_feature_metadata,
    expected_array,
    column,
    transpose,
):
    # given
    adata = _to_anndata(example_data)
    adata = at.pp.add_metadata(adata, example_sample_metadata, axis=0)
    adata = at.pp.add_metadata(adata, example_feature_metadata, axis=1)

    # when
    if not transpose:
        array = _adata_column_to_array(adata, column)
    else:
        array = _adata_column_to_array(adata.transpose(), column)

    # then
    assert np.all(array == expected_array)
