from importlib import metadata

from . import data, io, metrics, pl, pp, tl

__all__: list[str] = ["data", "io", "metrics", "pl", "pp", "tl"]

try:
    __version__ = metadata.version("alphapepttools")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"
