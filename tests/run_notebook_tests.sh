#!/bin/bash

# Run the notebook tests.

export IS_PYTEST_RUN=True

# TODO enable also 05_pycombat.ipynb
ALL_NBS=$(find ../docs/notebooks -name "*.ipynb" | grep -v "05_pycombat")

python -m pytest --nbmake $(echo $ALL_NBS)
