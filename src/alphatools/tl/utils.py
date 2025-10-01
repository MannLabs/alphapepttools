import contextlib

import matplotlib.pyplot as plt


# Context manager to suppress plots if needed
@contextlib.contextmanager
def _suppress_plots():  # noqa: ANN202 # avoid generator return type annotation
    original_show = plt.show
    plt.show = lambda *a, **k: None  # NOQA: ARG005
    try:
        yield
        plt.close("all")
    finally:
        plt.show = original_show
