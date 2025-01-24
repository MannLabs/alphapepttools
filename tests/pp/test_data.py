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
        smd = example_sample_metadata.copy()
        fmd = example_feature_metadata.copy()
    else:
        df = example_data.copy()
        smd = example_sample_metadata.copy().iloc[:2, :]  # keep samples cell1 and cell3
        fmd = example_feature_metadata.copy().iloc[:2, :]  # keep features G1 and G3

    # create AnnData object (this would already be done during data loading; here substituted with a private method)
    adata = _to_anndata(df)

    # add some existing metadata to check whether it is kept
    adata.obs = pd.DataFrame({"batch_new": ["11", "22", "33"]}, index=["cell1", "cell2", "cell3"])
    adata.var = pd.DataFrame({"UniProtID_new": ["P23456", "P34567", "P45678"]}, index=["G1", "G2", "G3"])

    # Add metadata to data
    # when
    adata = at.pp.add_metadata(
        adata, smd, axis=0, keep_data_shape=keep_data_shape, keep_existing_metadata=keep_existing_metadata
    )
    adata = at.pp.add_metadata(
        adata, fmd, axis=1, keep_data_shape=keep_data_shape, keep_existing_metadata=keep_existing_metadata
    )

    # check whether data was correctly added and aligned
    df_aligned = _get_df_from_adata(adata)
    smd_aligned = adata.obs
    fmd_aligned = adata.var

    # main tests for data, sample-, and feature metadata
    assert df_aligned.equals(expected_data), "Data should be aligned with sample and feature metadata"
    assert smd_aligned.equals(expected_sample_metadata), "Sample metadata should be aligned with data"
    assert fmd_aligned.equals(expected_feature_metadata), "Feature metadata should be aligned with data"

    # assert whether input data was changed
    assert df.equals(example_data), "Data should not be changed by adding it to Data object"
    assert smd.equals(example_sample_metadata if not keep_data_shape else example_sample_metadata.iloc[:2, :]), (
        "Sample metadata should not be changed by adding it to Data object"
    )
    assert fmd.equals(example_feature_metadata if not keep_data_shape else example_feature_metadata.iloc[:2, :]), (
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
