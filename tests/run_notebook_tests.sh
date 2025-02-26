#!/bin/bash

# Run the notebook tests.

# TODO enable also 03b_basic_workflow.ipynb
ALL_NBS=$(find ../notebooks -name "*.ipynb" | grep -v "03b_basic_workflow.ipynb")

python -m pytest --nbmake $(echo $ALL_NBS)
