import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.pp.impute import impute_gaussian


@pytest.fixture
def dummy_data_for_imputation():
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


def test_impute_gaussian_reproducibility(dummy_data_for_imputation):
    """Test that imputation with fixed random state produces reproducible results."""

    adata = dummy_data_for_imputation()
    adata_imputed = impute_gaussian(adata, std_offset=3, std_factor=0.3, random_state=0)

    # These values are hardcoded based on random_state=0, so we know what to expect
    # Compute manually once, then freeze here
    expected_A3 = np.random.RandomState(0).normal(
        loc=np.nanmean([1, 2, 4, 5]) - 3 * np.nanstd([1, 2, 4, 5]), scale=np.nanstd([1, 2, 4, 5]) * 0.3, size=1
    )[0]

    expected_B2 = np.random.RandomState(0).normal(
        loc=np.nanmean([10, 30, 40, 50]) - 3 * np.nanstd([10, 30, 40, 50]),
        scale=np.nanstd([10, 30, 40, 50]) * 0.3,
        size=1,
    )[0]

    imputed = adata_imputed.to_df()

    assert np.allclose(imputed.loc["s3", "A"], expected_A3)
    assert np.allclose(imputed.loc["s2", "B"], expected_B2)
    assert not np.isnan(imputed.loc["s3", "A"])
    assert not np.isnan(imputed.loc["s2", "B"])
