import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.pp.impute import impute_gaussian


@pytest.fixture
def gaussian_imputation_dummy_data():
    def create_data():
        data = pd.DataFrame(
            {
                "A": [1.0, 2.0, np.nan, 4.0, 5.0],
                "B": [10.0, np.nan, 30.0, 40.0, 50.0],
            },
            index=["s1", "s2", "s3", "s4", "s5"],
        )
        return ad.AnnData(data)

    return create_data()


def test_impute_gaussian(gaussian_imputation_dummy_data):
    """Test that imputation with fixed random state produces reproducible results."""

    RANDOM_STATE = 42
    STD_FACTOR = 0.3
    STD_OFFSET = 3
    A_VALS = [1, 2, 4, 5]
    B_VALS = [10, 30, 40, 50]

    adata_imputed = impute_gaussian(
        gaussian_imputation_dummy_data, std_offset=STD_OFFSET, std_factor=STD_FACTOR, random_state=RANDOM_STATE
    )

    rng = np.random.default_rng(RANDOM_STATE)

    expected_A3 = rng.normal(
        loc=np.nanmean(A_VALS) - STD_OFFSET * np.nanstd(A_VALS), scale=np.nanstd(A_VALS) * STD_FACTOR, size=1
    )[0]

    expected_B2 = rng.normal(
        loc=np.nanmean(B_VALS) - STD_OFFSET * np.nanstd(B_VALS),
        scale=np.nanstd(B_VALS) * STD_FACTOR,
        size=1,
    )[0]

    imputed = adata_imputed.to_df()

    assert np.allclose(imputed.loc["s3", "A"], expected_A3)
    assert np.allclose(imputed.loc["s2", "B"], expected_B2)
    assert not np.isnan(imputed.loc["s3", "A"])
    assert not np.isnan(imputed.loc["s2", "B"])
