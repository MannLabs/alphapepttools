"""Utility decorators for package"""

import functools
import inspect
import json
import re
import warnings
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import anndata as ad

from alphapepttools.settings import settings

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


def _serialize_args(arguments: dict[str, Any]) -> str:
    """Serialize bound call arguments to a JSON string

    Scalars are kept as they are; any other value, including `adata`, is reduced to its type name
    and, where available, its shape.
    """

    # break down complex objects into summaries
    def summarize(value: object) -> str | int | float | bool | None:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        shape = getattr(value, "shape", None)
        if shape is not None:
            return f"{type(value).__name__}{tuple(shape)}"
        return type(value).__name__

    return json.dumps({name: summarize(value) for name, value in arguments.items()}, default=repr)


def _record_provenance(func: Callable[..., Any], *, adata: object, result: object, args_json: str) -> None:
    """Append an entry for a call to `adata.uns["provenance"]` of the AnnData it produced or modified

    Parameters
    ----------
    func
        The called function
    adata
        Value of the call's `adata` argument, if any
    result
        Return value of the call
    args_json
        Serialized call arguments, see :func:`_serialize_args`
    """
    # avoid circular import
    from alphapepttools import __version__

    target = result if isinstance(result, ad.AnnData) else adata
    if not isinstance(target, ad.AnnData):
        return

    entry = {
        "function": f"{func.__module__}.{func.__qualname__}",
        "version": __version__,
        "args": args_json,
    }
    log = target.uns.setdefault("provenance", {})
    log[f"{len(log):04d}"] = entry


def track_provenance(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to record the provenance of a function call for an AnnData object.

    If provenance tracking is enabled via `settings.track_provenance`, this decorator will record the function call, its arguments and the current alphapepttools version in the `provenance` entry in `AnnData.uns`. Depending on the `inplace` behavior, the record will be either added directly to the input AnnData object or the returned copy.

    Tracking is disabled for nested calls, i.e. if a decorated function calls another decorated function, the inner call will not be recorded (i.e. if function XYZ uses pp.nanlog internally, there will be only a recording for the call to XYZ, not pp.nanlog).

    Functions decorated with this decorator should have an `adata` argument representing an AnnData object.

    Parameters
    ----------
    func
        Function taking an AnnData as argument `adata`, or returning one

    Returns
    -------
    The wrapped function

    Examples
    --------
    Entries are recorded by default and can be read back from `uns`:

    .. code-block:: python


        import alphapepttools as apt
        import anndata as ad
        import pandas as pd

        adata = ad.AnnData(
            pd.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": [4, 5, 6],
                }
            )
        )

        apt.pp.nanlog(adata)
        adata.uns["provenance"]["0000"]["function"]  # 'alphapepttools.pp.transform.nanlog'

    Switch tracking off for a block of code:

    .. code-block:: python

        with apt.settings.override(track_provenance=False):
            apt.pp.nanlog(adata)  # not recorded


    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        record = settings.track_provenance
        if record:
            # Serialized before the call so that `adata` is recorded with its input shape
            bound = inspect.signature(func).bind(*args, **kwargs)
            # Fill in args not provided by the caller, needed to track defaults
            bound.apply_defaults()
            args_json = _serialize_args(bound.arguments)
        with settings.override(track_provenance=False):
            result = func(*args, **kwargs)
        if record:
            _record_provenance(func, adata=bound.arguments.get("adata"), result=result, args_json=args_json)
        return result

    return wrapper
