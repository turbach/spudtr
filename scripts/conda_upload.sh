#!/bin/bash

# Intended for TravisCI deploy stage.
 
# For testing, can fake the TravisCI env like so
#   export TRAVIS="true"
#   export TRAVIS_BRANCH="X.Y.Z" 

if [[ "$TRAVIS" != "true" ]]
then
    echo "This is meant for a TravisCI build, do not run locally."
    exit -1
fi

# master branch builds get labeled for anacoda package "main"
# others labeled by branch for testing

if [[ $TRAVIS_BRANCH = "master" ]]; 
then
    conda_label="main"
else
    conda_label=br$TRAVIS_BRANCH  # anaconda -label chokes on leading digits
fi

echo "home: $HOME travis branch: $TRAVIS_BRANCH conda label: $conda_label"

# force convert b.c. of compiled C extension ... whatever works cross-platform works
rm -f -r ./tmp-conda-builds
mkdir -p ./tmp-conda-builds/linux-64
cp ${CONDA_PREFIX}/conda-bld/linux-64/spudtr-*.tar.bz2 ./tmp-conda-builds/linux-64
conda convert --platform all ${CONDA_PREFIX}/conda-bld/linux-64/spudtr-*.tar.bz2 --output-dir ./tmp-conda-builds --force
/bin/ls -l ./tmp-conda-builds/**/spudtr-*.tar.bz2


echo "Deploying to Anaconda.org like so ..."
conda_cmd="anaconda --token $ANACONDA_TOKEN upload ./tmp-conda-builds/**/spudtr-*.tar.bz2 -l $conda_label"
echo ${conda_cmd}
if ${conda_cmd};
then
    echo "Successfully deployed to Anaconda.org."
else
    echo "Error deploying to Anaconda.org"
    exit -3
fi
exit 0
