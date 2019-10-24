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

# spudtr/__version__
__version__ = get_ver()

# enforce conda meta.yaml semantic version is the same
jinja_version = f'{{% set version = "{__version__}" %}}'
meta_yaml_f = Path("./conda/meta.yaml")
with open(meta_yaml_f) as f:
    if not re.match(r"^" + jinja_version, f.read()):
        fail_msg = (
            "conda/meta.yaml must start with a jinja variable line exactly like this: "
            f"{jinja_version}"
        )
        raise Exception(fail_msg)


setup(
    name="spudtr",
    version=__version__,
    description="pandas dataframe function transforms",
    author="Thomas P. Urbach",
    author_email="turbach@ucsd.edu",
    url="http://kutaslab.ucsd.edu/people/urbach",
    packages=find_packages(),
    scripts=["bin/stub"],
    #    cmdclass={"build_ext": build_ext},
    #    ext_modules=cythonize(extensions, language_level=3),
)
