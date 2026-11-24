from scverse_misc.logging import TimedLogger
from scverse_misc.logging import get_logger as _scverse_logger

_LOGGER = _scverse_logger("alphapepttools", timed=True)


def get_logger() -> TimedLogger:
    """Return ``scverse.alphapepttools``, whose level follows :data:`scverse_misc.logging.config`."""
    return _LOGGER
