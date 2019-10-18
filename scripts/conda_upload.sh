#!/bin/bash

# For TravisCI deploy stage only

if [[ "$TRAVIS" != "true" ]]
then
    echo "This only works in a TravisCI build, do not try to run locally"
    exit -1
fi

# master branch builds get labeled for anacoda package "main"
# others labeled by branch for testing
if [[ $TRAVIS_BRANCH = "master" ]]; 
then
    conda_label="main"
else
    conda_label=$TRAVIS_BRANCH
fi

echo "home: $HOME travis branch: $TRAVIS_BRANCH conda label: $conda_label"

# spudtr is only linux64 
# echo "Converting conda package..."
# conda convert --platform all $HOME/miniconda/conda-bld/linux-64/spudtr-*.tar.bz2 --output-dir conda-build/

echo "Deploying to Anaconda.org..."
# anaconda -t $ANACONDA_TOKEN upload conda-build/**/spudtr-*.tar.bz2

# echo "Successfully deployed to Anaconda.org."
exit 0
