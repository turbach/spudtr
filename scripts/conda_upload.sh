#!usr/bin/bash

if [[ $TRAVIS_BRANCH == "master" ]]; 
then
    conda_label=main
else
    conda_label=$TRAVIS_BRANCH
fi

echo "travis branch: $TRAVIS_BRANCH conda label: $conda_label"
