#!/bin/bash

cp -r $RECIPE_DIR/../.. $SRC_DIR
$PYTHON -c "import pymbar"
$PYTHON setup.py install

# Add more build steps here, if they are necessary.

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
