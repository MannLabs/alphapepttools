from importlib import import_module as _import_module
from types import ModuleType as _ModuleType

__all__: list[str] = ["data", "io", "metrics", "pl", "pp", "tl"]

__version__ = "0.2.1-dev0"


def __getattr__(name: str) -> _ModuleType:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return _import_module(f".{name}", __name__)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
