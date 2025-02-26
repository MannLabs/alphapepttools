# TODO make tutorial_dev_spectral_libraries.ipynb work


ALL_NBS=$(find ../notebooks -name "*.ipynb")

python -m pytest --nbmake $(echo $ALL_NBS)
