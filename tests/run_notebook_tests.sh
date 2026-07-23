#!/bin/bash

# Run the notebook tests.

export IS_PYTEST_RUN=True

# study_03_biomarker_skin is excluded as the data is not publicly available yet
# study_04_scdvp is excluded as it includes decoupler as dependency for the downstream analysis, which is not needed for the package
# supplementary_02_scalability-demo_PCAn is a supplementary notebook and not part of tutorials/studies yet

# TODO: enable also study_03_biomarker_skin.ipynb
EXCLUDE_PATTERN="study_03_biomarker_skin|study_04_scDVP|supplementary_02_scalability-demo_PCAn"

ALL_NBS=$(find ../docs/notebooks -name "*.ipynb" | grep -vE "$EXCLUDE_PATTERN")

python -m pytest --nbmake $(echo $ALL_NBS)
