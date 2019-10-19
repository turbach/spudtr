#!/bin/bash

# This has some branch switching logic so master branch is uploaded to
# conda with the label "main" and non-master branches are uploaded
# with label "latest${TRAVIS_BRANCH} and clobber previous uploads on
# that branch. This is for round-trip develop, test, upload-to-conda,
# install-from-conda, test cycle
#
# For use with a .travis.yml deploy script provider after a fresh build 
# like so
# 
# # BEGIN .travis.yml ----------------------------------------
# 
# env:
#     - PACKAGE_NAME: spudtr   # used in deploy script
# language: minimal
# before_install:
#     - wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
#     - bash miniconda.sh -b -p $HOME/miniconda
#     - export PATH="$HOME/miniconda/bin:$PATH"
#     - hash -r
#     - conda config --set always_yes yes --set changeps1 no
#     - conda info -a
#     - conda install conda-build conda-verify
# install:
#     - conda build conda -c defaults -c conda-forge
#     # package setup here e.g., for instance ...
#     - conda create --name spudtr_env spudtr -c local -c defaults -c conda-forge
#     - source activate spudtr_env
#     - conda install black pytest-cov
#     - conda list
#     - lscpu
#     - python -c 'import numpy; numpy.show_config()'
#     - export TRAVIS_BRANCH  # for deploy script
#     - export HOME
#     - export TRAVIS
# script:
#     - black --check --verbose --line-length=79 .
#     - pytest --cov=spudtr
# after_success:
#     - pip install codecov && codecov
# before_deploy:
#     - pip install sphinx sphinx_rtd_theme jupyter nbsphinx nbconvert!=5.4
#     - conda install -c conda-forge pandoc
#     - conda install anaconda-client
#     - conda list
#     - sphinx-apidoc -e -f -o docs/source . ./tests/* ./setup.py
#     - make -C docs html
#     - touch docs/build/html/.nojekyll
#     - export ANACONDA_TOKEN
# deploy:
#     # master and working branches are packaged and labeled for conda
#     - provider: script
#       skip_cleanup: true
#       script: bash ./scripts/conda_upload.sh
#       on:
#           all_branches: true
#
# # END .travis.yml ----------------------------------------

#   NOTE: For testing locally in an active conda env, fake the
#     TravisCI env like so and make sure there is a unique
#     package tar.bz2 in the relevant conda-bld dir before converting.
#
#     export TRAVIS="true"
#     export TRAVIS_BRANCH="X.Y.Z" 
#     export ANACONDA_TOKEN="ku-actual-token ..."
#

if [ $USER = "travis" ]; then
    bld_prefix="/home/travis/miniconda"
else
    bld_prefix=${CONDA_PREFIX}
fi


if [ "$TRAVIS" != "true" ]; then
    echo "This is meant for a TravisCI build, do not run locally."
    exit -1
fi

# master branch builds get labeled for anacoda package "main"
# others labeled by branch for testing
if [ $TRAVIS_BRANCH = "master" ]; 
then
    # do *NOT* force master onto main, we want version collisions to fail
    FORCE="" 
    conda_label="main"
else
    # *DO* force non-master braches onto their label so we can conda install latest
    # for testing. Careful, this means non-master branches labeled like a released
    # X.Y.Z clobber main X.Y.Z
    FORCE="--force"  # 
    conda_label=latest$TRAVIS_BRANCH
fi

# thus far ...
echo "package name: $PACKAGE_NAME"
echo "conda prefix: $CONDA_PREFIX"
echo "travis branch: $TRAVIS_BRANCH"
echo "conda label: $conda_label"
echo "force flag: $FORCE"

# not needed for travis but tar.bz2 files can accumulate during local testing
bz_ls="/bin/ls -1 ${bld_prefix}/conda-bld/linux-64/${PACKAGE_NAME}-*.tar.bz2"
$bz_ls
n_bz2=`${bz_ls} | wc -l`
if (( $n_bz2 != 1 )); then
    echo "found ${n_bz2} ${bld_prefix}/conda-bld/linux-64/${PACKAGE_NAME}"'-*.tar.bz2'
    echo "there must be exactly one to convert for the conda upload"
    exit -2
fi

# force convert even thouhg compiled C extension ... whatever works cross-platform works
rm -f -r ./tmp-conda-builds
mkdir -p ./tmp-conda-builds/linux-64
cp ${bld_prefix}/conda-bld/linux-64/${PACKAGE_NAME}-*.tar.bz2 ./tmp-conda-builds/linux-64
conda convert --platform all ${bld_prefix}/conda-bld/linux-64/${PACKAGE_NAME}-*.tar.bz2 --output-dir ./tmp-conda-builds --force
/bin/ls -l ./tmp-conda-builds/**/${PACKAGE_NAME}-*.tar.bz2

echo "Deploying to Anaconda.org like so ..."
conda_cmd="anaconda --token $ANACONDA_TOKEN upload ./tmp-conda-builds/**/${PACKAGE_NAME}-*.tar.bz2 --label $conda_label --register ${FORCE}"
echo ${conda_cmd}

if ${conda_cmd};
then
    echo "Successfully deployed to Anaconda.org."
else
    echo "Error deploying to Anaconda.org"
    exit -3
fi
exit 0
