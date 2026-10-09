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
    def test_normalize_function_group_column(self, strategy, data_type, test_data_factory) -> None:
        """Test that groupwise normalization works"""
        X, expected_arrays, expected_factors = test_data_factory.get_test_data(data_type)

        # Generate an adata object with 3 groups
        # Adds a "batch" column in adata.obs
        BATCH_COLUMN = "batch"
        N_GROUPS = 3
        adata = ad.concat(
            {idx: ad.AnnData(X=X.copy() * (idx + 1)) for idx in range(N_GROUPS)},
            axis="obs",
            join="inner",
            label=BATCH_COLUMN,
            index_unique="-",  # Make observations unique across replicates
        )

        # Each group-wise array is the same  - concatenate expected values
        expected_result = np.concatenate([expected_arrays[strategy] * (idx + 1) for idx in range(N_GROUPS)], axis=0)
        expected_factors_tiled = np.tile(expected_factors[strategy], N_GROUPS)

        normalize(adata, strategy=strategy, group_column=BATCH_COLUMN, key_added="norm_factors")

        assert np.allclose(adata.X, expected_result, atol=1e-6, equal_nan=True)
        assert np.allclose(adata.obs["norm_factors"], expected_factors_tiled, atol=1e-6, equal_nan=True)

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
        """TMT dataset with 3 runs + 2 channels (no reference channel; per-group mean is used)"""
        # Per-group per-feature means: [1, 1], [1, 2], [1, 4]
        # Geometric mean across groups: feature 0 -> 1.0, feature 1 -> 2.0
        # Rows within each group are distinct so the test actually exercises averaging.
        X = np.array(
            [
                [0.5, 0.5],  # Run 0 - Sample 0
                [1.5, 1.5],  # Run 0 - Sample 1
                [0.5, 1.0],  # Run 1 - Sample 0
                [1.5, 3.0],  # Run 1 - Sample 1
                [0.5, 2.0],  # Run 2 - Sample 0
                [1.5, 6.0],  # Run 2 - Sample 1
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
        irs_data__reference_values: tuple[ad.AnnData, dict, np.ndarray],
        *,
        reference_kwargs,
        layer: str | None,
        copy: bool,
    ) -> None:
        """Test internal reference scaling"""
        adata, irs_kwargs, expected_array = irs_data__reference_values

        original_X = adata.X.copy()

        result = irs(adata, **reference_kwargs, **irs_kwargs, copy=copy, layer=layer)

        if copy:
            assert isinstance(result, ad.AnnData)
            modified_adata = result

        else:
            assert result is None
            modified_adata = adata

        modified_layer = modified_adata.X if layer is None else modified_adata.layers[layer]

        assert np.isclose(modified_layer, expected_array, atol=1e-6, equal_nan=True).all()

        if layer is not None:
            assert np.array_equal(modified_adata.X, original_X)

    @pytest.mark.parametrize("copy", [False, True])
    @pytest.mark.parametrize("layer", [None, "new_layer"])
    def test_irs__mean_reference(
        self,
        irs_data__mean_reference: tuple[ad.AnnData, dict, np.ndarray],
        *,
        layer: str | None,
        copy: bool,
    ) -> None:
        """Test internal reference scaling"""
        adata, irs_kwargs, expected_array = irs_data__mean_reference

        original_X = adata.X.copy()

        result = irs(adata, **irs_kwargs, copy=copy, layer=layer)

        if copy:
            assert isinstance(result, ad.AnnData)
            modified_adata = result

        else:
            assert result is None
            modified_adata = adata

        modified_layer = modified_adata.X if layer is None else modified_adata.layers[layer]

        assert np.isclose(modified_layer, expected_array, atol=1e-6, equal_nan=True).all()

        if layer is not None:
            assert np.array_equal(modified_adata.X, original_X)

    @pytest.mark.parametrize(
        "reference_kwargs",
        [
            {"reference_column": "tmt_channel", "reference_value": "does_not_exist"},
            {"reference_column": "is_reference", "reference_value": "does_not_exist"},
        ],
    )
    def test_irs__reference_value_missing(
        self, irs_data__reference_values: tuple[ad.AnnData, dict, np.ndarray], *, reference_kwargs
    ):
        """Test that function raises if reference value is not in column"""
        adata, irs_kwargs, _ = irs_data__reference_values

        with pytest.raises(ValueError, match="`reference_value` .* does not exist"):
            irs(adata, **reference_kwargs, **irs_kwargs, copy=False, layer=None)

    def test_irs__reference_value_missing_in_group(
        self, irs_data__reference_values: tuple[ad.AnnData, dict, np.ndarray]
    ):
        """Test that function raises if reference value is not available for one group"""
        adata, irs_kwargs, _ = irs_data__reference_values
        # reference missing for last group
        adata.obs["is_reference__one_missing"] = [
            False,
            True,
            False,
            True,
            False,
            False,
        ]

        with pytest.raises(ValueError, match=r"`reference_value` .* does not exist"):
            irs(
                adata,
                reference_column="is_reference__one_missing",
                reference_value=True,
                **irs_kwargs,
                copy=False,
                layer=None,
            )

    def test_irs__reference_column_without_value(self, irs_data__reference_values: tuple[ad.AnnData, dict, np.ndarray]):
        """Test that a `reference_column` without a `reference_value` warns and then finds no reference"""
        adata, irs_kwargs, _ = irs_data__reference_values

        with (
            pytest.warns(UserWarning, match="`reference_value` is None while `reference_column` is set"),
            pytest.raises(ValueError, match=r"`reference_value` .* does not exist"),
        ):
            irs(adata, reference_column="tmt_channel", **irs_kwargs, copy=False, layer=None)

    def test_irs__reference_value_without_column(self, irs_data__mean_reference: tuple[ad.AnnData, dict, np.ndarray]):
        """Test that a `reference_value` without a `reference_column` warns and is ignored"""
        adata, irs_kwargs, expected_array = irs_data__mean_reference
        # the fixture pins reference_column/reference_value to None; only the value is set here
        irs_kwargs = {**irs_kwargs, "reference_value": True}

        with pytest.warns(UserWarning, match="`reference_value` is set while `reference_column` is None"):
            result = irs(adata, **irs_kwargs, copy=True, layer=None)

        # ignoring `reference_value` means falling back to the per-group mean reference
        np.testing.assert_allclose(result.X, expected_array, atol=1e-6)
