import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphapepttools.pp import impute_gaussian, impute_knn, impute_median
from alphapepttools.pp.impute import _check_all_nan, _impute_knn, _impute_nanmedian


@pytest.fixture
def imputation_dummy_data() -> np.ndarray:
    """Test data for imputation methods"""
    # 4 x 5
    # Complete feature, complete feature, imputed feature, all nan
    return np.array(
        [
            [0.0, 0.0, 2.0, np.nan, 0.0],
            [1.0, 1.0, 3.0, 1.0, 10.0],
            [0.0, 2.0, 4.0, np.nan, 20.0],
            [np.nan, 3.0, 5.0, 3.0, np.nan],
        ]
    )


@pytest.fixture
def dummy_data_all_nan() -> np.ndarray:
    """Dummy data with a feature that only contains NaNs"""
    return np.array(
        [
            [0.0, 0.0, 2.0, np.nan, np.nan],
            [1.0, 1.0, 3.0, 1.0, np.nan],
            [0.0, 2.0, 4.0, np.nan, np.nan],
            [np.nan, 3.0, 5.0, 3.0, np.nan],
        ]
    )


@pytest.fixture
def median_imputation_dummy_data(imputation_dummy_data) -> tuple[np.ndarray, np.ndarray]:
    """Test data and reference for median imputation"""

    X_ref = np.array(
        [
            [0.0, 0.0, 2.0, 2.0, 0.0],
            [1.0, 1.0, 3.0, 1.0, 10.0],
            [0.0, 2.0, 4.0, 2.0, 20.0],
            [0.0, 3.0, 5.0, 3.0, 10.0],
        ]
    )

    return imputation_dummy_data, X_ref


@pytest.fixture
def knn_imputation_dummy_data(imputation_dummy_data) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Test data for median imputation"""
    X_ref = np.array(
        [
            [0.0, 0.0, 2.0, 2.25691494, 0.0],
            [1.0, 1.0, 3.0, 1.0, 10.0],
            [0.0, 2.0, 4.0, 2.67075186, 20.0],
            [0.33333333, 3.0, 5.0, 3.0, 16.66666667],
        ]
    )
    kwargs = {"n_neighbors": 2, "weights": "distance"}

    return imputation_dummy_data, X_ref, kwargs


def test___check_all_nan(dummy_data_all_nan) -> None:
    with pytest.raises(ValueError, match=r"Features with index \[4\]"):
        _check_all_nan(dummy_data_all_nan)


def test__impute_nanmedian(median_imputation_dummy_data) -> None:
    """Test median imputation for data with nan values"""
    X, X_ref = median_imputation_dummy_data

    X_imputed = _impute_nanmedian(X)

    assert np.all(np.isclose(X_imputed, X_ref, equal_nan=True))


def test__impute_knn(knn_imputation_dummy_data) -> None:
    """Test knn imputation for data with nan values"""
    X, X_ref, kwargs = knn_imputation_dummy_data

    X_imputed = _impute_knn(X, **kwargs)

    assert np.all(np.isclose(X_imputed, X_ref, equal_nan=True))


class TestImputeGaussian:
    @pytest.fixture
    def gaussian_imputation_dummy_data(self):
        def create_data():
            data = pd.DataFrame(
                {
                    "A": [1.0, 2.0, np.nan, 4.0, 5.0],
                    "B": [10.0, np.nan, 30.0, 40.0, 50.0],
                },
                index=["s1", "s2", "s3", "s4", "s5"],
            )
            return ad.AnnData(data, layers={"new_layer": data})

        return create_data()

    @pytest.mark.parametrize("copy", [False, True])
    @pytest.mark.parametrize("layer", [None, "new_layer"])
    def test_impute_gaussian(self, gaussian_imputation_dummy_data: ad.AnnData, layer: str, *, copy: bool) -> None:
        """Test that imputation with fixed random state produces reproducible results."""

        RANDOM_STATE = 42
        STD_FACTOR = 0.3
        STD_OFFSET = 3
        A_VALS = [1, 2, 4, 5]
        B_VALS = [10, 30, 40, 50]

        result = impute_gaussian(
            gaussian_imputation_dummy_data,
            std_offset=STD_OFFSET,
            std_factor=STD_FACTOR,
            random_state=RANDOM_STATE,
            layer=layer,
            copy=copy,
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

        adata_imputed = result if copy else gaussian_imputation_dummy_data

        imputed = adata_imputed.to_df(layer=layer)

        assert np.allclose(imputed.loc["s3", "A"], expected_A3)
        assert np.allclose(imputed.loc["s2", "B"], expected_B2)
        assert not np.isnan(imputed.loc["s3", "A"])
        assert not np.isnan(imputed.loc["s2", "B"])


class TestImputeMedianAnnData:
    @pytest.fixture
    def median_imputation_dummy_anndata(
        self,
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
                [0.0, 0.0, 2.0, 1.0, 0.0],
                [1.0, 1.0, 3.0, 1.0, 10.0],
                [0.0, 2.0, 4.0, 3.0, 20.0],
                [0.0, 3.0, 5.0, 3.0, 20.0],
            ]
        )

        return ad.AnnData(X, obs=obs, layers={"layer2": X}), X_ref, X_ref_grouped

    @pytest.fixture
    def median_imputation_dummy_anndata_all_nan(self, dummy_data_all_nan: np.ndarray) -> ad.AnnData:
        """AnnData object with a feature that contains only NaNs"""

        obs = pd.DataFrame(
            {
                "sample_id": ["A", "B", "C", "D"],
                "sample_group": ["A", "A", "B", "B"],
                "sample_group_with_nan": ["A", "A", np.nan, np.nan],
            }
        )

        return ad.AnnData(X=dummy_data_all_nan, obs=obs)

    @pytest.mark.parametrize("copy", [False, True])
    @pytest.mark.parametrize(
        ("layer", "group_column"),
        [(None, None), ("layer2", None), (None, "sample_group"), ("layer2", "sample_group")],
    )
    def test_impute_median(self, median_imputation_dummy_anndata, layer: str, group_column: str, *, copy: bool) -> None:
        """Test median imputation for data with nan values"""
        adata, X_ref, X_ref_grouped = median_imputation_dummy_anndata
        result = impute_median(adata, layer=layer, group_column=group_column, copy=copy)

        if copy:
            assert isinstance(result, ad.AnnData)
            adata_imputed = result
        else:
            assert result is None
            adata_imputed = adata
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

    @pytest.mark.parametrize("group_column", [None, "sample_group"])
    def test_impute_median__feature_all_nan(self, median_imputation_dummy_anndata_all_nan, group_column: str) -> None:
        """Test median imputation raises if a feature contains all nan"""
        adata = median_imputation_dummy_anndata_all_nan

        with pytest.raises(ValueError, match=r"Features with index \[4\]"):
            _ = impute_median(adata, group_column=group_column)

    def test_impute_median__raises_if_group_column_contains_nan(self, median_imputation_dummy_anndata) -> None:
        """Test that median imputation raises warning if group_column contains nan"""

        adata, _, _ = median_imputation_dummy_anndata

        with pytest.raises(ValueError, match="`group_column`"):
            _ = impute_median(adata, layer=None, group_column="sample_group_with_nan")

    def test_impute_median__missing_group_column(
        self,
        median_imputation_dummy_anndata,
    ) -> None:
        """Test that KeyError is raised if `group_column` does not exist in `adata.obs`"""
        adata, _, _ = median_imputation_dummy_anndata

        with pytest.raises(KeyError):
            impute_median(adata, group_column="non_existent_column")

    def test_impute_median__missing_layer(
        self,
        median_imputation_dummy_anndata,
    ) -> None:
        """Test that KeyError is raised if `layer` does not exist in `adata`"""
        adata, _, _ = median_imputation_dummy_anndata

        with pytest.raises(KeyError):
            impute_median(adata, layer="non_existent_layer")


class TestImputeKNNAnnData:
    @pytest.fixture
    def knn_imputation_dummy_anndata(
        self,
        knn_imputation_dummy_data,
    ) -> tuple[ad.AnnData, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
        """Test data for median imputation"""
        obs = pd.DataFrame(
            {
                "sample_id": ["A", "B", "C", "D"],
                "sample_group": ["A", "A", "B", "B"],
                "sample_group_with_nan": ["A", "A", np.nan, np.nan],
            }
        )

        X, X_ref, kwargs = knn_imputation_dummy_data
        X_ref_grouped = np.array(
            [
                [0.0, 0.0, 2.0, 1.0, 0.0],
                [1.0, 1.0, 3.0, 1.0, 10.0],
                [0.0, 2.0, 4.0, 3.0, 20.0],
                [0.0, 3.0, 5.0, 3.0, 20.0],
            ]
        )

        return ad.AnnData(X, obs=obs, layers={"layer2": X}), X_ref, X_ref_grouped, kwargs

    @pytest.fixture
    def knn_imputation_dummy_anndata_all_nan(self, dummy_data_all_nan: np.ndarray) -> ad.AnnData:
        """AnnData object with a feature that contains only NaNs"""

        obs = pd.DataFrame(
            {
                "sample_id": ["A", "B", "C", "D"],
                "sample_group": ["A", "A", "B", "B"],
                "sample_group_with_nan": ["A", "A", np.nan, np.nan],
            }
        )

        return ad.AnnData(X=dummy_data_all_nan, obs=obs)

    @pytest.mark.parametrize("copy", [True, False])
    @pytest.mark.parametrize("layer", [None, "layer2"])
    @pytest.mark.parametrize("group_column", [None, "sample_group"])
    def test_impute_knn(self, knn_imputation_dummy_anndata, layer: str, group_column: str, *, copy: bool) -> None:
        """Test median imputation for data with nan values"""
        adata, X_ref, X_ref_grouped, kwargs = knn_imputation_dummy_anndata

        result = impute_knn(adata, layer=layer, group_column=group_column, **kwargs, copy=copy)

        if copy:
            assert isinstance(result, ad.AnnData)
            adata_imputed = result
        else:
            assert result is None
            adata_imputed = adata

        X_imputed = adata_imputed.X if layer is None else adata_imputed.layers[layer]

        if group_column is None:
            assert np.all(np.isclose(X_imputed, X_ref, equal_nan=True))
        elif group_column == "sample_group":
            assert np.all(np.isclose(X_imputed, X_ref_grouped, equal_nan=True))
        else:
            pytest.fail("Unexpected group column passed")

    @pytest.mark.parametrize("group_column", [None, "sample_group"])
    def test_impute_median__feature_all_nan(self, knn_imputation_dummy_anndata_all_nan, group_column: str) -> None:
        """Test median imputation raises if a feature contains all nan"""
        adata = knn_imputation_dummy_anndata_all_nan

        with pytest.raises(ValueError, match=r"Features with index"):
            _ = impute_median(adata, group_column=group_column)

    def test_impute_median__raises_if_group_column_contains_nan(self, knn_imputation_dummy_anndata_all_nan) -> None:
        """Test that median imputation raises warning if group_column contains nan"""

        adata, _, _, _ = knn_imputation_dummy_anndata_all_nan

        with pytest.raises(ValueError, match="Features with index"):
            _ = impute_knn(adata, layer=None, group_column="sample_group_with_nan", n_neighbors=1, copy=True)

    def test_impute_median__missing_group_column(
        self,
        knn_imputation_dummy_anndata_all_nan,
    ) -> None:
        """Test that KeyError is raised if `group_column` does not exist in `adata.obs`"""
        adata, _, _, _ = knn_imputation_dummy_anndata_all_nan

        with pytest.raises(KeyError):
            _ = impute_knn(adata, group_column="non_existent_column", n_neighbors=1, copy=True)

    def test_impute_median__missing_layer(
        self,
        knn_imputation_dummy_anndata_all_nan,
    ) -> None:
        """Test that KeyError is raised if `layer` does not exist in `adata`"""
        adata, _, _, _ = knn_imputation_dummy_anndata_all_nan

        with pytest.raises(KeyError):
            _ = impute_knn(adata, layer="non_existent_layer", copy=True)
