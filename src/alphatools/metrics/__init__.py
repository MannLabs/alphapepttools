"""Metrics for the quality assessment of the analysis"""

from ._pmad import pmad
from ._principal_component_regression import principal_component_regression

__all__ = ["pmad", "principal_component_regression"]
