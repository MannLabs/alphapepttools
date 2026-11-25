from importlib import metadata

from . import data, io, metrics, pl, pp, tl
from ._settings import settings

try:
    # Hard-coded __version__ tags have been dropped in favor of git tags.
    # Use `git describe --tags` to read the current version in the git repository.
    __version__ = metadata.version("alphapepttools")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__: list[str] = ["__version__", "data", "io", "metrics", "pl", "pp", "settings", "tl"]
