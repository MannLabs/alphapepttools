import pytest

from alphapepttools._utils import _resolve_axis


class TestResolveAxis:
    @pytest.mark.parametrize(("axis", "expected"), [("obs", "obs"), (0, "obs"), ("var", "var"), (1, "var")])
    def test_valid(self, axis, expected):
        assert _resolve_axis(axis) == expected

    @pytest.mark.parametrize("bad_axis", ["invalid", 2, -1, "0", None])
    def test_invalid(self, bad_axis):
        with pytest.raises(ValueError, match="axis must be 'obs', 'var', 0, or 1"):
            _resolve_axis(bad_axis)
