from typing import Literal

import pytest
from alphabase.pg_reader import pg_reader_provider
from alphabase.psm_reader import psm_reader_provider

from alphapepttools.io import list_available_reader


@pytest.mark.parametrize("kind", ["psm_reader", "pg_reader"])
def test_available_reader(kind: Literal["psm_reader", "pg_reader"]) -> None:
    list_of_available_reader = list_available_reader(kind)

    if kind == "psm_reader":
        assert len(list_of_available_reader) == len(psm_reader_provider.reader_dict)
    elif kind == "pg_reader":
        assert len(list_of_available_reader) == len(pg_reader_provider.reader_dict)
    assert "alphadia" in list_of_available_reader
