import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.pp.transform import check_data_integrity, nanlog

# Suppress warnings about conversion to numpy arrays: inputs are ints and get
# converted to floats by AnnData upon storing the dataframe.
pytestmark = pytest.mark.filterwarnings("ignore:.*converted to numpy array.*")


# Test log transformation
@pytest.fixture
def log_dummy_data():
    def create_data():
        # Create a DataFrame with some dummy data
        data = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [-1, np.nan, np.inf, -np.inf, 10000],
            },
            index=["a", "b", "c", "d", "e"],
        )
        return ad.AnnData(data)

    return create_data()


@pytest.mark.parametrize(
    ("base", "input_type", "expected"),
    [
        (
            2,
            ad.AnnData,
            ad.AnnData(
                pd.DataFrame(
                    {"A": [np.log2(x) for x in [1, 2, 3, 4, 5]], "B": [np.nan, np.nan, np.nan, np.nan, np.log2(10000)]},
                    index=["a", "b", "c", "d", "e"],
                )
            ),
        ),
        (
            10,
            ad.AnnData,
            ad.AnnData(
                pd.DataFrame(
                    {
                        "A": [np.log10(x) for x in [1, 2, 3, 4, 5]],
                        "B": [np.nan, np.nan, np.nan, np.nan, np.log10(10000)],
                    },
                    index=["a", "b", "c", "d", "e"],
                )
            ),
        ),
        (
            3.5,
            ad.AnnData,
            ad.AnnData(
                pd.DataFrame(
                    {
                        "A": [np.log(x) / np.log(3.5) for x in [1, 2, 3, 4, 5]],
                        "B": [np.nan, np.nan, np.nan, np.nan, np.log(10000) / np.log(3.5)],
                    },
                    index=["a", "b", "c", "d", "e"],
                )
            ),
        ),
        (
            2,
            pd.DataFrame,
            pd.DataFrame(
                {"A": [np.log2(x) for x in [1, 2, 3, 4, 5]], "B": [np.nan, np.nan, np.nan, np.nan, np.log2(10000)]},
                index=["a", "b", "c", "d", "e"],
            ),
        ),
        (
            10,
            pd.DataFrame,
            pd.DataFrame(
                {"A": [np.log10(x) for x in [1, 2, 3, 4, 5]], "B": [np.nan, np.nan, np.nan, np.nan, np.log10(10000)]},
                index=["a", "b", "c", "d", "e"],
            ),
        ),
        (
            3.5,
            pd.DataFrame,
            pd.DataFrame(
                {
                    "A": [np.log(x) / np.log(3.5) for x in [1, 2, 3, 4, 5]],
                    "B": [np.nan, np.nan, np.nan, np.nan, np.log(10000) / np.log(3.5)],
                },
                index=["a", "b", "c", "d", "e"],
            ),
        ),
        (
            2,
            pd.Series,
            [
                pd.Series([np.log2(x) for x in [1, 2, 3, 4, 5]], index=["a", "b", "c", "d", "e"], name="A"),
                pd.Series([np.nan, np.nan, np.nan, np.nan, np.log2(10000)], index=["a", "b", "c", "d", "e"], name="B"),
            ],
        ),
        (
            10,
            pd.Series,
            [
                pd.Series([np.log10(x) for x in [1, 2, 3, 4, 5]], index=["a", "b", "c", "d", "e"], name="A"),
                pd.Series([np.nan, np.nan, np.nan, np.nan, np.log10(10000)], index=["a", "b", "c", "d", "e"], name="B"),
            ],
        ),
        (
            3.5,
            pd.Series,
            [
                pd.Series(
                    [np.log(x) / np.log(3.5) for x in [1, 2, 3, 4, 5]], index=["a", "b", "c", "d", "e"], name="A"
                ),
                pd.Series(
                    [np.nan, np.nan, np.nan, np.nan, np.log(10000) / np.log(3.5)],
                    index=["a", "b", "c", "d", "e"],
                    name="B",
                ),
            ],
        ),
        (
            2,
            np.ndarray,
            np.array([[np.log2(x) for x in [1, 2, 3, 4, 5]], [np.nan, np.nan, np.nan, np.nan, np.log2(10000)]]),
        ),
        (
            10,
            np.ndarray,
            np.array([[np.log10(x) for x in [1, 2, 3, 4, 5]], [np.nan, np.nan, np.nan, np.nan, np.log10(10000)]]),
        ),
        (
            3.5,
            np.ndarray,
            np.array(
                [
                    [np.log(x) / np.log(3.5) for x in [1, 2, 3, 4, 5]],
                    [np.nan, np.nan, np.nan, np.nan, np.log(10000) / np.log(3.5)],
                ]
            ),
        ),
    ],
)
def test_nanlog(log_dummy_data, base, input_type, expected):
    """Test nanlog function with different input types and log bases."""

    # AnnData
    if input_type == ad.AnnData:
        result = nanlog(log_dummy_data, base)
        pd.testing.assert_frame_equal(result.to_df(), expected.to_df())
    # Pandas DataFrame
    elif input_type == pd.DataFrame:
        result = nanlog(log_dummy_data.to_df(), base)
        pd.testing.assert_frame_equal(result, expected)
    # Pandas Series
    elif input_type == pd.Series:
        result = [nanlog(log_dummy_data.to_df()[col], base) for col in log_dummy_data.to_df().columns]
        for res, exp in zip(result, expected, strict=False):
            pd.testing.assert_series_equal(res, exp)
    # Numpy ndarray
    else:
        result = [nanlog(log_dummy_data.to_df()[col].to_numpy(), base) for col in log_dummy_data.to_df().columns]
        for res, exp in zip(result, expected, strict=False):
            np.testing.assert_array_equal(res, exp)


@pytest.mark.parametrize(
    ("base", "input", "expected_error"),
    [
        (0, np.array([0, 1]), ValueError),
        (1, np.array([0, 1]), ValueError),
        (-1, np.array([0, 1]), ValueError),
        (2, "invalid_input", TypeError),
    ],
)
def test_nanlog_errors(base, input_data, expected_error):
    """Test nanlog function for invalid base values."""
    with pytest.raises(expected_error):
        nanlog(input_data, base)


@pytest.mark.parametrize(
    ("input", "expected_mask", "verbosity", "expect_warnings"),
    [
        (
            np.array([np.nan, 0, -1, np.inf, -np.inf, 1, 2.1]),
            np.array([True, True, True, True, True, False, False]),
            0,
            False,
        ),
        (
            np.array([np.nan, 0, -1, np.inf, -np.inf, 1, 2.1]),
            np.array([True, True, True, True, True, False, False]),
            1,
            True,
        ),
    ],
)
def test_check_data_integrity(input_data, expected_mask, verbosity, expect_warnings, caplog):
    """Test check_data_integrity function."""

    if not expect_warnings:
        result_mask = check_data_integrity(input_data, verbosity=verbosity)
        np.testing.assert_array_equal(result_mask, expected_mask)
    else:
        with caplog.at_level("WARNING"):
            result_mask = check_data_integrity(input_data, verbosity=verbosity)

        assert "1 nan values in the data." in caplog.text
        assert "1 zero values in the data." in caplog.text
        assert "2 negative values in the data." in caplog.text
        assert "1 inf values in the data." in caplog.text
        assert "1 negative_inf values in the data." in caplog.text
