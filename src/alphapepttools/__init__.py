from importlib import metadata

from . import data, io, metrics, pl, pp, tl
from .settings import settings

__all__: list[str] = ["data", "io", "metrics", "pl", "pp", "settings", "tl"]

try:
    # Hard-coded __version__ tags have been dropped in favor of git tags.
    # Use `git describe --tags` to read the current version in the git repository.
    __version__ = metadata.version("alphapepttools")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"
