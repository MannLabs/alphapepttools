"""Utility decorators for package"""

import functools
import re
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

# Matches a numpydoc parameter entry `inplace` (optionally `inplace : type`) and its indented description.
# Indentation is captured rather than assumed, since Python>=3.13 dedents `__doc__` at compile time.
_INPLACE_PARAM_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)inplace(?:[ \t]*:[^\n]*)?\n(?:(?P=indent)[ \t]+\S[^\n]*\n)+",
    flags=re.MULTILINE,
)
"""Regex to match an inplace parameter in a numpydoc-docstring"""


def _add_copy_to_docstring(docstring: str, *, qualname: str, since: str, removed_in: str) -> str:
    def append_copy_entry(match: re.Match) -> str:
        indent = match.group("indent")
        return (
            f"{match.group(0)}"
            f"{indent}copy\n"
            f"{indent}    .. deprecated:: {since}\n"
            f"{indent}        Use `inplace` instead. `copy` will be removed in version {removed_in}.\n"
        )

    updated_docstring, n_replacements = _INPLACE_PARAM_PATTERN.subn(append_copy_entry, docstring, count=1)
    if n_replacements == 0:
        raise ValueError(f"Docstring of `{qualname}` does not document the `inplace` argument")
    return updated_docstring


def deprecated_copy(*, since: str = "0.5.0", removed_in: str = "0.6.0") -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Deprecate the `copy` argument of a function in favor of `inplace`

    The decorated function must define a keyword-only argument `inplace` and must not define `copy`.
    The wrapper accepts `copy` as an additional keyword argument: if a caller sets it, a `FutureWarning`
    is emitted and the value is forwarded as `inplace = not copy`. A deprecation notice for `copy`
    is added to the docstring after the `inplace` entry.

    Parameters
    ----------
    since
        Version in which `copy` was deprecated
    removed_in
        Version in which `copy` will be removed

    Returns
    -------
    Decorator that wraps the function

    Raises
    ------
    ValueError
        At decoration time, if the docstring does not document `inplace`.
        At call time, if both `copy` and `inplace` are passed with contradicting values
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            copy = kwargs.pop("copy", None)
            if copy is not None:
                warnings.warn(
                    f"The `copy` argument of `{func.__qualname__}` is deprecated since version {since} "
                    f"and will be removed in version {removed_in}. Use `inplace={not copy}` instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                # Checking the kwargs rather than the value detects an explicit `inplace` without
                # requiring a `None` sentinel default in the function signature
                if "inplace" in kwargs and kwargs["inplace"] == copy:
                    raise ValueError(
                        f"Contradicting arguments `copy={copy}` and `inplace={kwargs['inplace']}`. Only pass `inplace`."
                    )
                kwargs["inplace"] = not copy
            return func(*args, **kwargs)

        if func.__doc__ is not None:
            wrapper.__doc__ = _add_copy_to_docstring(
                func.__doc__, qualname=func.__qualname__, since=since, removed_in=removed_in
            )

        return wrapper

    return decorator
