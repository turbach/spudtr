# to work in development mode use:  pip install -e .

# from Cython.Distutils import build_ext
# from Cython.Build import cythonize
# import numpy as np

from setuptools import find_packages, setup  # , Extension
from pathlib import Path
import re
from spudtr import get_ver

# extensions = [
#      Extension("_spudtr", ["src/_spudtr.pyx"], include_dirs=[np.get_include()])
# ]

__version__ = get_ver()

# enforce conda meta.yaml and __init__.py version are the same
jinja_version = f'{{% set version = "{__version__}" %}}'
meta_yaml_f = Path("./conda/meta.yaml")
with open(meta_yaml_f) as f:
    if not re.match(r"^" + jinja_version, f.read()):
        fail_msg = (
            "conda/meta.yaml must start with a jinja variable line exactly like this: "
            f"{jinja_version}"
        )
        raise Exception(fail_msg)

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
    package_data={"spudtr": ["data/gh_*"]},
    #    scripts=["bin/stub"],
    #    cmdclass={"build_ext": build_ext},
    #    ext_modules=cythonize(extensions, language_level=3),
)
