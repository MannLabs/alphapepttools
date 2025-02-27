#!/bin/bash

# Run the notebook tests.

export IS_PYTEST_RUN=True

ALL_NBS=$(find ../notebooks -name "*.ipynb")

python -m pytest --nbmake $(echo $ALL_NBS)
