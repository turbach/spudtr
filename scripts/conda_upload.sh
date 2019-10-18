#!/bin/bash

# Intended for TravisCI deploy stage.
 
# For testing, can fake the TravisCI env like so
#   export TRAVIS="true"
#   export TRAVIS_BRANCH="X.Y.Z" 

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

# no conda convert b.c. of compiled C extension ... so only support linux-64 for now
# conda convert --platform all $HOME/miniconda/conda-bld/linux-64/spudtr-*.tar.bz2 --output-dir conda-build/

if [ ! -f $HOME/miniconda/conda-bld/linux-64/spudtr-*.tar.bz2 ]; 
then
    echo "file not found " '$HOME/miniconda/conda-bld/linux-64/spudtr-*.tar.bz2'
    exit -2
else
    ls -l $HOME/miniconda/conda-bld/linux-64/spudtr-*.tar.bz2
fi

echo "Deploying to Anaconda.org like so ..."
echo "anaconda -t $ANACONDA_TOKEN upload $HOME/miniconda/conda-bld/linux-64/spudtr-*.tar.bz2 --label $conda_label"
if anaconda -t $ANACONDA_TOKEN upload $HOME/miniconda/conda-bld/linux-64/spudtr-*.tar.bz2 --label $conda_label;
then
    echo "Successfully deployed to Anaconda.org."
else
    echo "Error deploying to Anaconda.org"
    exit -3
fi
exit 0
