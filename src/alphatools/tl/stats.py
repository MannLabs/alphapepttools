# Statistics functionalities for working with AnnData objects

import logging

# logging configuration
logging.basicConfig(level=logging.INFO)


def ttest() -> None:
    raise NotImplementedError


def pca() -> None:
    raise NotImplementedError


# Automatically define __all__ to contain public names
__all__: list[str] = [name for name in globals() if not name.startswith("_")]
