import anndata as ad
import numpy as np
import pytest

from alphatools.pp import normalize
from alphatools.pp.norm import _total_mean_normalization, _total_median_normalization, _validate_strategies


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
        assert len(adata.layers) == 0

    @pytest.mark.parametrize("strategy", ["total_mean", "total_median"])
    @pytest.mark.parametrize("data_type", ["all_equal", "different", "nan"])
    def test_normalize_function_key_added(self, strategy, data_type, test_data_factory) -> None:
        X, expected_arrays, _ = test_data_factory.get_test_data(data_type)
        adata = ad.AnnData(X=X.copy())
        normalize(adata, strategy=strategy, key_added="norm_factors")

        assert np.isclose(adata.X, expected_arrays[strategy], atol=1e-6, equal_nan=True).all()
        assert len(adata.obs.columns) == 1
        assert "norm_factors" in adata.obs.columns

    @pytest.mark.parametrize("strategy", ["total_mean", "total_median"])
    @pytest.mark.parametrize(
        "from_layer",
        [None, "different_layer"],
    )
    @pytest.mark.parametrize("to_layer", [None, "new_layer", "different_layer"])
    def test_normalize_function_layer_operations(
        self, strategy: str, from_layer: str, to_layer: str, test_data_factory
    ) -> None:
        X, expected_arrays, expected_norm_factors = test_data_factory.get_test_data("different")

        # Construct multi-layered anndata
        layers = {from_layer: X.copy()} if from_layer is not None else None
        adata = ad.AnnData(X=X.copy(), layers=layers)

        normalize(adata, strategy=strategy, key_added="norm_factors", from_layer=from_layer, to_layer=to_layer)

        result = adata.X if to_layer is None else adata.layers[to_layer]

        assert np.isclose(result, expected_arrays[strategy], atol=1e-6, equal_nan=True).all()
        assert "norm_factors" in adata.obs.columns
        assert np.isclose(adata.obs["norm_factors"], expected_norm_factors[strategy], atol=1e-6, equal_nan=True).all()

    @pytest.mark.parametrize("strategy", ["total_mean", "total_median"])
    def test_normalize_function_to_layer_exists(self, strategy: str, test_data_factory) -> None:
        X, expected_arrays, _ = test_data_factory.get_test_data("different")
        to_layer = "to_layer_exists"
        # Construct multi-layered anndata
        adata = ad.AnnData(X=X.copy(), layers={to_layer: X.copy()})

        normalize(adata, strategy=strategy, key_added="norm_factors", to_layer=to_layer)

        assert np.isclose(adata.layers[to_layer], expected_arrays[strategy], atol=1e-6, equal_nan=True).all()
