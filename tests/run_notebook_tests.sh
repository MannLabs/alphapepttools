#!/bin/bash

# Run the notebook tests.

export IS_PYTEST_RUN=True

ALL_NBS=$(find ../docs/notebooks -name "*.ipynb" | grep -v -e "study_03_biomarker_skin" -e "study_04_scDVP.ipynb" -e "study_02_peptidomics_pelsa.ipynb")


python -m pytest --nbmake $(echo $ALL_NBS)
