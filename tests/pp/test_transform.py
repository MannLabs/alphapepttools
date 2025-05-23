import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.pp.transform import nanlog

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
    ("log", "input_type", "expected"),
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
            2,
            np.ndarray,
            np.array([[np.log2(x) for x in [1, 2, 3, 4, 5]], [np.nan, np.nan, np.nan, np.nan, np.log2(10000)]]),
        ),
        (
            10,
            np.ndarray,
            np.array([[np.log10(x) for x in [1, 2, 3, 4, 5]], [np.nan, np.nan, np.nan, np.nan, np.log10(10000)]]),
        ),
    ],
)
def test_nanlog(log_dummy_data, log, input_type, expected):
    """Test nanlog function with different input types and log bases."""

    def _warning_free_to_df(adata):
        return pd.DataFrame(adata.X.copy(), index=adata.obs_names, columns=adata.var_names)

    if input_type == ad.AnnData:
        result = nanlog(log_dummy_data, log)
        pd.testing.assert_frame_equal(_warning_free_to_df(result), _warning_free_to_df(expected))
    elif input_type == pd.DataFrame:
        result = nanlog(log_dummy_data.to_df(), log)
        pd.testing.assert_frame_equal(result, expected)
    elif input_type == pd.Series:
        result = [nanlog(log_dummy_data.to_df()[col], log) for col in log_dummy_data.to_df().columns]
        for res, exp in zip(result, expected, strict=False):
            pd.testing.assert_series_equal(res, exp)
    elif input_type == np.ndarray:
        result = [nanlog(log_dummy_data.to_df()[col].values, log) for col in log_dummy_data.to_df().columns]
        for res, exp in zip(result, expected, strict=False):
            np.testing.assert_array_equal(res, exp)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")
