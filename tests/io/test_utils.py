from typing import Literal

import pytest
from alphabase.pg_reader import pg_reader_provider
from alphabase.psm_reader import psm_reader_provider

from alphapepttools.io import available_reader


@pytest.mark.parametrize("reader_type", ["psm_reader", "pg_reader"])
def test_available_reader(reader_type: Literal["psm_reader", "pg_reader"]) -> None:
    list_of_available_reader = available_reader(reader_type)

    if reader_type == "psm_reader":
        assert len(list_of_available_reader) == len(psm_reader_provider.reader_dict)
    elif reader_type == "pg_reader":
        assert len(list_of_available_reader) == len(pg_reader_provider.reader_dict)
    assert "alphadia" in list_of_available_reader
