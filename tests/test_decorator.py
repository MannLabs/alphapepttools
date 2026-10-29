import re
import warnings

import pytest

from alphapepttools._decorator import deprecated_copy

SINCE = "0.5.0"
REMOVED_IN = "0.6.0"


def _func(data: list, *, inplace: bool = True) -> bool:
    """Dummy function

    Parameters
    ----------
    data
        Input data
    inplace
        Whether to modify inplace.
        Second line of description
    other
        Other parameter

    Returns
    -------
    Received `inplace` value
    """
    return inplace


@pytest.fixture
def decorated_func():
    return deprecated_copy(since=SINCE, removed_in=REMOVED_IN)(_func)


class TestDeprecatedCopyCall:
    def test_default_call_does_not_warn(self, decorated_func) -> None:
        """Calls without `copy` must not emit a deprecation warning"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = decorated_func([1])

        assert result is True

    @pytest.mark.parametrize("inplace", [True, False])
    def test_inplace_only_does_not_warn(self, decorated_func, *, inplace: bool) -> None:
        """Calls with only `inplace` are forwarded unchanged and do not warn"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = decorated_func([1], inplace=inplace)

        assert result is inplace

    @pytest.mark.parametrize(
        ("kwargs", "expected_inplace"),
        [
            ({"copy": True}, False),
            ({"copy": False}, True),
            ({"copy": True, "inplace": False}, False),
            ({"copy": False, "inplace": True}, True),
        ],
    )
    def test_maps_copy_to_inplace(self, decorated_func, kwargs: dict, *, expected_inplace: bool) -> None:
        """Explicit `copy` warns and is forwarded as `inplace = not copy` without reaching the function"""
        with pytest.warns(FutureWarning, match=rf"`copy`.*deprecated.*{SINCE}.*{REMOVED_IN}"):
            result = decorated_func([1], **kwargs)

        assert result is expected_inplace

    @pytest.mark.parametrize("value", [True, False])
    def test_raises_on_contradicting_arguments(self, decorated_func, *, value: bool) -> None:
        """Contradicting `copy` and `inplace` raise"""
        with pytest.warns(FutureWarning), pytest.raises(ValueError, match="Contradicting"):
            decorated_func([1], copy=value, inplace=value)

    def test_preserves_metadata(self, decorated_func) -> None:
        """Wrapper exposes the metadata of the wrapped function"""
        assert decorated_func.__name__ == _func.__name__
        assert decorated_func.__wrapped__ is _func


class TestDeprecatedCopyDocstring:
    def test_adds_copy_after_inplace(self, decorated_func) -> None:
        """A deprecated `copy` entry is inserted directly after the complete `inplace` entry"""
        docstring = decorated_func.__doc__

        assert f".. deprecated:: {SINCE}" in docstring
        assert f"removed in version {REMOVED_IN}" in docstring
        assert re.search(r"Second line of description\n[ \t]*copy\n", docstring)
        assert docstring.index("copy\n") < docstring.index("other\n")

    def test_raises_on_undocumented_inplace(self) -> None:
        """Decorating a function whose docstring lacks `inplace` raises"""

        def func(*, inplace: bool = True) -> None:
            """Parameters
            ----------
            other
                Other parameter
            """

        decorator = deprecated_copy(since=SINCE, removed_in=REMOVED_IN)

        with pytest.raises(ValueError, match="does not document"):
            decorator(func)
