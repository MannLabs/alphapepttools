import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphapepttools.pp import irs, normalize
from alphapepttools.pp.norm import _total_mean_normalization, _total_median_normalization, _validate_strategies


class TestDataFactory:
    """Factory for generating test data for normalization tests"""

    @staticmethod
    def get_test_data(data_type: str) -> tuple[np.ndarray, dict, dict]:
        """Get test data based on type"""
        data_configs = {
            "all_equal": {
                "X": np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]]),
                "expected_arrays": {
                    "total_mean": np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]]),
                    "total_median": np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]]),
                },
                "expected_factors": {
                    "total_mean": np.array([1.0, 1.0, 1.0]),
                    "total_median": np.array([1.0, 1.0, 1.0]),
                },
            },
            "different": {
                "X": np.array([[0.8, 1.0], [2.0, 0.0], [0.0, 2.0]]),
                "expected_arrays": {
                    "total_mean": np.array([[0.85925926, 1.07407407], [1.93333333, 0.0], [0.0, 1.93333333]]),
                    "total_median": np.array([[0.88888889, 1.11111111], [2.0, 0.0], [0.0, 2.0]]),
                },
                "expected_factors": {
                    "total_mean": np.array([1.07407407, 0.96666667, 0.96666667]),
                    "total_median": np.array([1.111111, 1.0, 1.0]),
                },
            },
            "nan": {
                "X": np.array([[0, 1.0], [2.0, 0.0], [0.8, np.nan], [np.nan, 2.0]]),
                "expected_arrays": {
                    "total_mean": np.array([[0, 1.45], [1.45, 0.0], [1.45, np.nan], [np.nan, 1.45]]),
                    "total_median": np.array([[0.0, 1.5], [1.5, 0.0], [1.5, np.nan], [np.nan, 1.5]]),
                },
                "expected_factors": {
                    "total_mean": np.array([1.45, 0.725, 1.8124999, 0.725]),
                    "total_median": np.array([1.5, 0.75, 1.875, 0.75]),
                },
            },
        }

        config = data_configs[data_type]
        return config["X"], config["expected_arrays"], config["expected_factors"]


@pytest.fixture
def test_data_factory():
    return TestDataFactory()


@pytest.fixture
def nan_count_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate count data (samples, features) where samples have different intensities and contain nan values,
    the expected result and the expected normalization factors"""
    # Add nan intensites
    return (
        np.array([[0, 1.0], [2.0, 0.0], [0.8, np.nan], [np.nan, 2.0]]),
        {
            "total_mean": np.array([[0, 1.45], [1.45, 0.0], [1.45, np.nan], [np.nan, 1.45]]),
            "total_median": np.array([[0.0, 1.5], [1.5, 0.0], [1.5, np.nan], [np.nan, 1.5]]),
        },
        {"total_mean": np.array([1.45, 0.725, 1.8124999, 0.725]), "total_median": np.array([1.5, 0.75, 1.875, 0.75])},
    )


class TestValidation:
    """Test input validation"""

    @pytest.mark.parametrize("strategy", ["total_mean", "total_median"])
    def test__validate_strategies(self, strategy) -> None:
        # Valid strategy
        _validate_strategies(strategy)

    def test__valid_strategies_invalid(self) -> None:
        # Invalid strategy
        with pytest.raises(ValueError, match="`strategy` must be one of"):
            _validate_strategies("invalid_strategy")


class TestNormalizationFunctions:
    """Test low-level normalization functions"""

    @pytest.mark.parametrize("strategy", ["total_mean", "total_median"])
    @pytest.mark.parametrize("data_type", ["all_equal", "different", "nan"])
    def test_normalization_function(self, strategy, data_type, test_data_factory) -> None:
        """Test correctness of normalization function"""
        X, expected_arrays, expected_factors = test_data_factory.get_test_data(data_type)

        norm_func = {"total_mean": _total_mean_normalization, "total_median": _total_median_normalization}[strategy]

        norm_array, norm_factors = norm_func(X)

        assert np.isclose(norm_array, expected_arrays[strategy], atol=1e-6, equal_nan=True).all()
        assert np.isclose(norm_factors, expected_factors[strategy], atol=1e-6, equal_nan=True).all()


class TestNormalizeFunction:
    """Test the high-level normalize function"""

    @pytest.mark.parametrize("strategy", ["total_mean", "total_median"])
    @pytest.mark.parametrize("data_type", ["all_equal", "different", "nan"])
    def test_normalize_function_default(self, strategy, data_type, test_data_factory) -> None:
        X, expected_arrays, _ = test_data_factory.get_test_data(data_type)
        adata = ad.AnnData(X=X.copy())
        normalize(adata, strategy=strategy)

        assert np.isclose(adata.X, expected_arrays[strategy], atol=1e-6, equal_nan=True).all()
        assert len(adata.obs.columns) == 0
        assert [k for k in adata.layers if k is not None] == []

    @pytest.mark.parametrize("strategy", ["total_mean", "total_median"])
    @pytest.mark.parametrize("data_type", ["all_equal", "different", "nan"])
    def test_normalize_function_key_added(self, strategy, data_type, test_data_factory) -> None:
        X, expected_arrays, _ = test_data_factory.get_test_data(data_type)
        adata = ad.AnnData(X=X.copy())
        normalize(adata, strategy=strategy, key_added="norm_factors")

        assert np.isclose(adata.X, expected_arrays[strategy], atol=1e-6, equal_nan=True).all()
        assert len(adata.obs.columns) == 1
        assert "norm_factors" in adata.obs.columns

    @pytest.mark.parametrize("copy", [False, True])
    @pytest.mark.parametrize("layer", [None, "new_layer", "different_layer"])
    @pytest.mark.parametrize("strategy", ["total_mean", "total_median"])
    def test_normalize_function_layer_operations(
        self,
        strategy: str,
        layer: str,
        copy: bool,  # noqa: FBT001
        test_data_factory,
    ) -> None:
        """Test that results are stored in the correct layers"""
        X, expected_arrays, expected_norm_factors = test_data_factory.get_test_data("different")

        # Construct multi-layered anndata
        layers = {layer: X.copy()} if layer is not None else None
        adata = ad.AnnData(X=X.copy(), layers=layers)

        result = normalize(adata, strategy=strategy, key_added="norm_factors", layer=layer, copy=copy)

        if copy:
            assert isinstance(result, ad.AnnData)
            modified_adata = result

        else:
            assert result is None
            modified_adata = adata

        modified_layer = modified_adata.X if layer is None else modified_adata.layers[layer]

        assert np.isclose(modified_layer, expected_arrays[strategy], atol=1e-6, equal_nan=True).all()
        assert "norm_factors" in modified_adata.obs.columns
        assert np.isclose(
            modified_adata.obs["norm_factors"], expected_norm_factors[strategy], atol=1e-6, equal_nan=True
        ).all()


class TestIRS:
    @pytest.fixture
    def irs_data__reference_values(self):
        """TMT dataset with 3 runs + 2 channels"""
        # geometric mean of feature 0 in reference channels: 1.0
        # geometric mean of feature 1 in reference channels: 2.0
        X = np.array(
            [
                [0, 0],  # Run 0 - Sample
                [1, 1],  # Run 0 - Ref
                [1, 1],  # Run 1 - Sample
                [1, 2],  # Run 1 - Ref
                [2, 2],  # Run 2 - Sample
                [1, 4],  # Run 2 - Ref
            ],
            dtype=np.float32,
        )

        obs = pd.DataFrame(
            {
                "tmt_plex": [0, 0, 1, 1, 2, 2],
                "tmt_channel": [0, 1, 0, 1, 0, 1],
                "is_reference": [False, True, False, True, False, True],
            }
        )

        norm_factors = np.array([[1, 2], [1, 2], [1, 1], [1, 1], [1, 0.5], [1, 0.5]])

        ref = X * norm_factors

        irs_kwargs = {
            "group_column": "tmt_plex",
        }

        return ad.AnnData(X=X, obs=obs, layers={"new_layer": X.copy()}), irs_kwargs, ref

    @pytest.fixture
    def irs_data__mean_reference(self):
        """TMT dataset with 3 runs + 2 channels"""
        # geometric mean of feature 0 in reference channels: 1.0
        # geometric mean of feature 1 in reference channels: 2.0
        X = np.array(
            [
                [1, 1],  # Run 0 - Sample 0
                [1, 1],  # Run 0 - Sample 1
                [1, 2],  # Run 1 - Sample 0
                [1, 2],  # Run 1 - Sample 1
                [1, 4],  # Run 2 - Sample 0
                [1, 4],  # Run 2 - Sample 1
            ],
            dtype=np.float32,
        )

        obs = pd.DataFrame(
            {
                "tmt_plex": [0, 0, 1, 1, 2, 2],
                "tmt_channel": [0, 1, 0, 1, 0, 1],
                "is_reference": [False, False, False, False, False, False],
            }
        )

        norm_factors = np.array([[1, 2], [1, 2], [1, 1], [1, 1], [1, 0.5], [1, 0.5]])

        ref = X * norm_factors

        irs_kwargs = {
            "group_column": "tmt_plex",
            "reference_column": None,
            "reference_value": None,
        }

        return ad.AnnData(X=X, obs=obs, layers={"new_layer": X.copy()}), irs_kwargs, ref

    @pytest.mark.parametrize("copy", [False, True])
    @pytest.mark.parametrize("layer", [None, "new_layer"])
    @pytest.mark.parametrize(
        "reference_kwargs",
        [
            {"reference_column": "tmt_channel", "reference_value": 1},
            {"reference_column": "is_reference", "reference_value": True},
        ],
    )
    def test_irs__reference_values(
        self,
        irs_data__reference_values: tuple[ad.AnnData, np.ndarray],
        *,
        reference_kwargs,
        layer: str | None,
        copy: bool,
    ) -> None:
        """Test internal reference scaling"""
        adata, irs_kwargs, expected_array = irs_data__reference_values

        adata = adata.copy()

        assert isinstance(adata, ad.AnnData)

        result = irs(adata, **reference_kwargs, **irs_kwargs, copy=copy, layer=layer)

        if copy:
            assert isinstance(result, ad.AnnData)
            modified_adata = result

        else:
            assert result is None
            modified_adata = adata

        modified_layer = modified_adata.X if layer is None else modified_adata.layers[layer]

        assert np.isclose(modified_layer, expected_array, atol=1e-6, equal_nan=True).all()

    @pytest.mark.parametrize("copy", [False, True])
    @pytest.mark.parametrize("layer", [None, "new_layer"])
    def test_irs__mean_reference(
        self, irs_data__mean_reference: tuple[ad.AnnData, np.ndarray], *, layer: str | None, copy: bool
    ) -> None:
        """Test internal reference scaling"""
        adata, irs_kwargs, expected_array = irs_data__mean_reference

        adata = adata.copy()

        assert isinstance(adata, ad.AnnData)

        result = irs(adata, **irs_kwargs, copy=copy, layer=layer)

        if copy:
            assert isinstance(result, ad.AnnData)
            modified_adata = result

        else:
            assert result is None
            modified_adata = adata

        modified_layer = modified_adata.X if layer is None else modified_adata.layers[layer]

        assert np.isclose(modified_layer, expected_array, atol=1e-6, equal_nan=True).all()
