# to work in development mode use:  pip install -e .

from setuptools import find_packages, setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from pathlib import Path
import numpy as np
import re

extensions = [
    Extension("_spudtr", ["src/_spudtr.pyx"], include_dirs=[np.get_include()])
]

# single source versioning with a bit of error checking
def get_ver():
    with open(Path(__file__).parent / "spudtr/__init__.py", "r") as stream:
        pf_ver = re.search(
            r".*__version__.*=.*[\"\'](?P<ver_str>\d+\.\d+\.\d+\S*)[\'\"].*",
            stream.read(),
        )

    if pf_ver is None:
        msg = f"""
        spudtr __init__.py must have an X.Y.Z semantic version, e.g.,

        __version__ = '0.0.0'
        __version__ = '0.0.0.dev0.0'

        """
        raise ValueError(msg)
    else:
        return pf_ver["ver_str"]


setup(
    name="spudtr",
    version=get_ver(),
    description="pandas dataframe function transforms",
    author="Thomas P. Urbach",
    author_email="turbach@ucsd.edu",
    url="http://kutaslab.ucsd.edu/people/urbach",
    packages=find_packages(),
    scripts=["bin/stub"],
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions, language_level=3),
)
