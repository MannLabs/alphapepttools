#!/bin/bash

# Run the notebook tests.

export IS_PYTEST_RUN=True

# TODO enable also study_03_biomarker_skin.ipynb
EXCLUDE_PATTERN="study_03_biomarker_skin"

# Check if inmoose is available, exclude study_02_peptidomics_pelsa if not
if ! python -c "import inmoose" 2>/dev/null; then
    EXCLUDE_PATTERN="study_03_biomarker_skin|study_02_peptidomics_pelsa"
fi

ALL_NBS=$(find ../docs/notebooks -name "*.ipynb" | grep -vE "$EXCLUDE_PATTERN")

python -m pytest --nbmake $(echo $ALL_NBS)
