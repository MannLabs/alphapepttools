import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.pp.transform import detect_special_values, nanlog

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
    ("base", "expected"),
    [
        (
            2,
            ad.AnnData(
                pd.DataFrame(
                    {"A": [np.log2(x) for x in [1, 2, 3, 4, 5]], "B": [np.nan, np.nan, np.nan, np.nan, np.log2(10000)]},
                    index=["a", "b", "c", "d", "e"],
                )
            ),
        ),
        (
            10,
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
    ],
)
def test_nanlog(log_dummy_data, base, expected):
    """Test nanlog function with different input types and log bases."""

    # AnnData
    result = nanlog(log_dummy_data, base)
    pd.testing.assert_frame_equal(result.to_df(), expected.to_df())


@pytest.mark.parametrize(
    ("base", "input_data", "expected_error"),
    [
        (0, ad.AnnData(np.array([[0, 1]])), ValueError),
        (1, ad.AnnData(np.array([[0, 1]])), ValueError),
        (-1, ad.AnnData(np.array([[0, 1]])), ValueError),
        (2, "invalid_input", TypeError),
    ],
)
def test_nanlog_errors(base, input_data, expected_error):
    """Test nanlog function for invalid base values."""
    with pytest.raises(expected_error):
        nanlog(input_data, base)


@pytest.mark.parametrize(
    ("input_data", "expected_mask", "verbosity", "expect_warnings"),
    [
        # 1D array with various invalid values without warnings
        (
            np.array([np.nan, 0, -1, np.inf, -np.inf, 1, 2.1]),
            np.array([True, True, True, True, True, False, False]),
            0,
            False,
        ),
        # 2D array with various invalid values without warnings
        (
            np.array([[np.nan, 0, -1, np.inf, -np.inf, 1, 2.1]]),
            np.array([[True, True, True, True, True, False, False]]),
            0,
            False,
        ),
        # 1D array with various invalid values with warnings
        (
            np.array([np.nan, 0, -1, np.inf, -np.inf, 1, 2.1]),
            np.array([True, True, True, True, True, False, False]),
            1,
            True,
        ),
        # 2D array with various invalid values with warnings
        (
            np.array([[np.nan, 0, -1, np.inf, -np.inf, 1, 2.1]]),
            np.array([[True, True, True, True, True, False, False]]),
            1,
            True,
        ),
    ],
)
def test_check_data_integrity(input_data, expected_mask, verbosity, expect_warnings, caplog):
    """Test check_data_integrity function."""

    if not expect_warnings:
        result_mask = detect_special_values(input_data, verbosity=verbosity)
        np.testing.assert_array_equal(result_mask, expected_mask)
    else:
        with caplog.at_level("WARNING"):
            result_mask = detect_special_values(input_data, verbosity=verbosity)

        assert "1 nan values in the data." in caplog.text
        assert "1 zero values in the data." in caplog.text
        assert "2 negative values in the data." in caplog.text
        assert "1 inf values in the data." in caplog.text
        assert "1 negative_inf values in the data." in caplog.text
