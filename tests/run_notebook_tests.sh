#!/bin/bash

# Run the notebook tests.

export IS_PYTEST_RUN=True

# study_03_biomarker_skin is excluded as the data is not publicly available yet
# TODO: enable also study_03_biomarker_skin.ipynb
EXCLUDE_PATTERN="study_03_biomarker_skin"

ALL_NBS=$(find ../docs/notebooks -name "*.ipynb" | grep -vE "$EXCLUDE_PATTERN")

python -m pytest --nbmake $(echo $ALL_NBS)
