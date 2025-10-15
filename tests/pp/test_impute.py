import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.pp.impute import _impute_nanmedian, impute_gaussian, impute_median


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


@pytest.fixture
def median_imputation_dummy_data() -> tuple[np.ndarray, np.ndarray]:
    """Test data for median imputation"""

    # 4 x 3
    # Complete feature, complete feature, imputed feature, all nan
    X = np.array(
        [
            [0.0, 0.0, 2.0, np.nan, np.nan],
            [1.0, 1.0, 3.0, 1.0, np.nan],
            [0.0, 2.0, 4.0, np.nan, np.nan],
            [np.nan, 3.0, 5.0, 3.0, np.nan],
        ]
    )
    X_ref = np.array(
        [
            [0.0, 0.0, 2.0, 2.0, np.nan],
            [1.0, 1.0, 3.0, 1.0, np.nan],
            [0.0, 2.0, 4.0, 2.0, np.nan],
            [0.0, 3.0, 5.0, 3.0, np.nan],
        ]
    )

    return X, X_ref


@pytest.fixture
def median_imputation_dummy_anndata(
    median_imputation_dummy_data,
) -> tuple[ad.AnnData, np.ndarray, np.ndarray, np.ndarray]:
    """Test data for median imputation"""
    obs = pd.DataFrame(
        {
            "sample_id": ["A", "B", "C", "D"],
            "sample_group": ["A", "A", "B", "B"],
            "sample_group_with_nan": ["A", "A", np.nan, np.nan],
        }
    )

    X, X_ref = median_imputation_dummy_data
    X_ref_grouped = np.array(
        [
            [0.0, 0.0, 2.0, 1.0, np.nan],
            [1.0, 1.0, 3.0, 1.0, np.nan],
            [0.0, 2.0, 4.0, 3.0, np.nan],
            [0.0, 3.0, 5.0, 3.0, np.nan],
        ]
    )
    X_ref_grouped_with_nan = np.array(
        [
            [0.0, 0.0, 2.0, 1.0, np.nan],
            [1.0, 1.0, 3.0, 1.0, np.nan],
            [0.0, 2.0, 4.0, np.nan, np.nan],
            [np.nan, 3.0, 5.0, 3.0, np.nan],
        ]
    )
    return ad.AnnData(X, obs=obs, layers={"layer2": X}), X_ref, X_ref_grouped, X_ref_grouped_with_nan


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


def test__impute_nanmedian(median_imputation_dummy_data) -> None:
    """Test median imputation for data with nan values"""
    X, X_ref = median_imputation_dummy_data

    with pytest.warns(UserWarning, match="Will remain nan."):
        X_imputed = _impute_nanmedian(X)

    assert np.all(np.isclose(X_imputed, X_ref, equal_nan=True))


@pytest.mark.parametrize(
    ("layer", "group_column"),
    [(None, None), ("layer2", None), (None, "sample_group"), ("layer2", "sample_group")],
)
def test_impute_median(median_imputation_dummy_anndata, layer: str, group_column: str) -> None:
    """Test median imputation for data with nan values"""
    adata, X_ref, X_ref_grouped, _ = median_imputation_dummy_anndata

    adata_imputed = impute_median(
        adata,
        layer=layer,
        group_column=group_column,
    )

    X_imputed = adata_imputed.X if layer is None else adata_imputed.layers[layer]

    if group_column is None:
        assert np.all(np.isclose(X_imputed, X_ref, equal_nan=True))
    elif group_column == "sample_group":
        assert np.all(np.isclose(X_imputed, X_ref_grouped, equal_nan=True))
    else:
        pytest.fail("Unexpected group column passed")


def test_impute_median__warns_if_group_column_contains_nan(median_imputation_dummy_anndata) -> None:
    """Test that median imputation raises warning if group_column contains nan"""

    adata, _, _, X_ref_grouped_with_nan = median_imputation_dummy_anndata

    with pytest.warns(UserWarning, match="`group_column`"):
        adata_imputed = impute_median(adata, layer=None, group_column="sample_group_with_nan")

    X_imputed = adata_imputed.X

    assert np.all(np.isclose(X_imputed, X_ref_grouped_with_nan, equal_nan=True))


def test_impute_median__missing_group_column(
    median_imputation_dummy_anndata,
) -> None:
    """Test that KeyError is raised if `group_column` does not exist in `adata.obs`"""
    adata, _, _, _ = median_imputation_dummy_anndata

    with pytest.raises(KeyError):
        impute_median(adata, group_column="non_existent_column")


def test_impute_median__missing_layer(
    median_imputation_dummy_anndata,
) -> None:
    """Test that KeyError is raised if `layer` does not exist in `adata`"""
    adata, _, _, _ = median_imputation_dummy_anndata

    with pytest.raises(KeyError):
        impute_median(adata, layer="non_existent_layer")
