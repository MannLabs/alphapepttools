import pytest

from alphapepttools._utils import resolve_axis


class TestResolveAxis:
    @pytest.mark.parametrize(("axis", "expected"), [("obs", 0), (0, 0), ("var", 1), (1, 1)])
    def test_valid(self, axis, expected):
        assert resolve_axis(axis) == expected

    @pytest.mark.parametrize("bad_axis", ["invalid", 2, -1, "0", None])
    def test_invalid(self, bad_axis):
        with pytest.raises(ValueError, match="axis must be 'obs', 'var', 0, or 1"):
            resolve_axis(bad_axis)
