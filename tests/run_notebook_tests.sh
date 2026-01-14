#!/bin/bash

# Run the notebook tests.

export IS_PYTEST_RUN=True

# TODO enable also study_03_biomarker_skin.ipynb
ALL_NBS=$(find ../docs/notebooks -name "*.ipynb" | grep -v "study_03_biomarker_skin")

python -m pytest --nbmake $(echo $ALL_NBS)
