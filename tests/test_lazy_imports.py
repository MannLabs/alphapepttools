import subprocess
import sys

_LAZY_IMPORT_CHECK = """
import sys
import alphapepttools

assert {f"alphapepttools.{name}" for name in alphapepttools.__all__}.isdisjoint(sys.modules)

pl = alphapepttools.pl
from alphapepttools.pl.figure import AxisManager

assert {"anndata", "scanpy", "alphapepttools.pl.plots"}.isdisjoint(sys.modules)

barplot = pl.barplot
assert barplot is pl.barplot
assert pl.plots is sys.modules["alphapepttools.pl.plots"]
"""


def test_lightweight_imports_are_lazy() -> None:
    subprocess.run([sys.executable, "-c", _LAZY_IMPORT_CHECK], check=True)  # noqa: S603
