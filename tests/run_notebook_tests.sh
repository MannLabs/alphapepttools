#!/bin/bash

# Run the notebook tests.

export IS_PYTEST_RUN=True

# TODO enable also study_03_biomarker_skin.ipynb
ALL_NBS=$(find ../docs/notebooks -name "*.ipynb" | grep -v -e "study_03_biomarker_skin" -e "study_04_scDVP.ipynb")

python -m pytest --nbmake $(echo $ALL_NBS)
