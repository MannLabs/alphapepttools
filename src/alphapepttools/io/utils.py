"""Utility functions for readers"""

from typing import Literal

from alphabase.pg_reader import pg_reader_provider
from alphabase.psm_reader import psm_reader_provider


def list_available_reader(kind: Literal["psm_reader", "pg_reader"] = "pg_reader") -> list[str]:
    """Get a list of all available readers, as provided by alphabase

    Parameters
    ----------
    kind
        Whether to return readers for peptice spectrum matches (`psm_reader`) or protein group
        intensities (`pg_reader`)

    Returns
    -------
    list[str]
        A list of all available readers that are accepted by the respective io function

    Example
    -------

    .. code-block:: python

        at.io.available_reader(reader_type="pg_reader")
        > ['alphadia', 'alphapept', ...]

    """
    if kind == "psm_reader":
        return sorted(psm_reader_provider.reader_dict.keys())
    if kind == "pg_reader":
        return sorted(pg_reader_provider.reader_dict.keys())
    raise KeyError(f"Pass either `psm_reader` or `pg_reader`, not {kind}")
