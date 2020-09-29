# to work in development mode use:  pip install -e .

from setuptools import find_packages, setup  # , Extension
from pathlib import Path
import re
from spudtr import get_ver

__version__ = get_ver()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="spudtr",
    version=__version__,
    description="some pandas utility dataframe transforms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Thomas P. Urbach",
    author_email="turbach@ucsd.edu",
    url="https://github.com/kutaslab/spudtr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
    ],
    packages=find_packages(exclude=["tests"]),
    scripts=["bin/stub"],
)
