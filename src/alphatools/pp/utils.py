"""Utility functions with functionality shared across modules"""

import functools
import inspect
from collections.abc import Callable
from typing import Any

import anndata as ad
import numpy as np

AnnDataFunc = Callable[[ad.AnnData, Any], ad.AnnData]
CopyAnnDataFunc = Callable[[ad.AnnData, Any], None | ad.AnnData]


def add_function_parameter_to_signature(
    func: Callable, parameter: str, kind: inspect._ParameterKind = inspect.Parameter.KEYWORD_ONLY, **kwargs
) -> Callable:
    """Add a parameter to a function signature

    Parameters
    ----------
    func
        Function that should be modified
    parameter
        Name of parameter
    **kwargs
        Passed to `inspect.Parameter`

    Returns
    -------
    Callable
        Updated function with copy parameter in signature
    """
    function_signature = inspect.signature(func)
    params = list(function_signature.parameters.values())

    copy_param = inspect.Parameter(parameter, kind=kind, **kwargs)
    new_params = (*params, copy_param)
    func.__signature__ = function_signature.replace(parameters=new_params)

    return func


def copy_decorator(func: AnnDataFunc) -> CopyAnnDataFunc:
    """Add copy functionality to function that modifies `anndata.AnnData` objects

    Parameters
    ----------
    func
        Callable that takes anndata objects and other parameters and returns an updated anndata object.
        First function argument to be named `adata` of type :class:`anndata.AnnData`.

    Returns
    -------
    Callable[[ad.AnnData, Any], None | ad.AnnData]
        Callable with added copy argument and updated function signature.

    Usage
    -----
    .. code-block:: python
        @copy_decorator
        def set_to_zero(adata: ad.AnnData) -> ad.AnnData | None:
            adata.X = np.zeros_like(adata.X)
            return adata

        help(set_to_zero)
        > set_to_zero(adata: anndata._core.anndata.AnnData, *, copy: bool = False) -> anndata._core.anndata.AnnData | None
    """
    COPY_DEFAULT = False

    @functools.wraps(func)
    def wrapper(adata: ad.AnnData, *args, copy: bool = COPY_DEFAULT, **kwargs) -> None | ad.AnnData:
        adata = adata.copy() if copy else adata
        result = func(adata, *args, **kwargs)
        return result if copy else None

    # Add copy to function signature
    return add_function_parameter_to_signature(
        wrapper, parameter="copy", kind=inspect.Parameter.KEYWORD_ONLY, default=COPY_DEFAULT, annotation=bool
    )


def get_anndata_layer(adata: ad.AnnData, layer: str | None = None) -> np.ndarray:
    """Get a layer from an anndata object

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` object
    layer
        Name of layer. If None (default), returns adata.X

    Returns
    -------
    np.ndarray | ArrayLike
        Array corresponding to the respective layer in the anndata object or adata.X if `layer=None`
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError(f"Expected an anndata.AnnData object, got {type(adata)}")

    if (layer is not None) and (layer not in adata.layers):
        raise KeyError(
            f"Layer {layer} is not in anndata. Please specify `None` for adata.X or an existing layer {adata.layers.keys()}"
        )

    return adata.X if layer is None else adata.layers[layer]


def update_adata_layer(adata: ad.AnnData, data: np.ndarray, layer: str | None = None) -> None:
    """Set data to a layer in an anndata object

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` object
    layer
        Name of layer. The layer might already exist which overwrites the current values.
        If None (default), updates adata.X

    Returns
    -------
    None
        The adata object is modified inplace

    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError(f"Expected an anndata.AnnData object, got {type(adata)}")

    if layer is None:
        adata.X = data
    else:
        adata.layers[layer] = data
