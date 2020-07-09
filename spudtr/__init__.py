from pathlib import Path
import re

DATA_DIR = Path(__file__).parents[0] / "data"

# EEG files for testing and docs in DATA_DIR
P3_F = "gh_sub000p3.epochs.h5"
P5_F = "gh_sub000p5.epochs.h5"
WR_F = "gh_sub000wr.epochs.h5"


# single source the python package version with a bit of error checking
__version__ = "0.0.6"


def get_ver():
    pf_ver = re.search(r"(?P<ver_str>\d+\.\d+\.\d+\S*)", __version__)

    if pf_ver is None:
        msg = f"""Illegal spudtr __version__: {__version__}
        spudtr __init__.py must have an X.Y.Z semantic version, e.g.,

        __version__ = '0.0.0'
        __version__ = '0.0.0.dev0'
        __version__ = '0.0.0rc1'

        """
        raise Exception(msg)
    else:
        return pf_ver["ver_str"]
