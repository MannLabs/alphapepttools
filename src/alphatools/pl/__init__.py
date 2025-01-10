from .basic import BasicClass, basic_plot
from .plots import *

#control wildcard imports
__all__ = []
from .plots import __all__ as _plots_all
__all__ += _plots_all
