# Tools for data processing

import logging

# logging configuration
logging.basicConfig(level=logging.INFO)


def umap() -> None:
    raise NotImplementedError


# Automatically define __all__ to contain public names
__all__: list[str] = [name for name in globals() if not name.startswith("_")]
