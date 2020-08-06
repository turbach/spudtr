from pathlib import Path
import re

DATA_DIR = Path(__file__).parents[0] / "data"
DATA_URL = r"https://zenodo.org/record/3968485/files/"

# EEG files for testing and docs in DATA_DIR
P3_F = "gh_sub000p3.epochs.h5"
P5_F = "gh_sub000p5.epochs.h5"
WR_F = "gh_sub000wr.epochs.h5"


# single source the python package version with a bit of error checking
__version__ = "0.0.9.dev0"


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


def get_demo_df(filename, url=DATA_URL):
    """fetch feather format demo EEG epochs data

    Parameters
    ----------
    filename : str
       file to fetch

    url : str default=eeg-workshops/mkpy_data_examples/data, v0.0.3
       top-level URL to fetch from

    Returns
    -------
    df : pandas.DataFrame
        spudtr epochs format data frame with epoch_id, time_ms columns

    """

    import pandas as pd
    import io
    import os
    import requests  # URL IO

    if not url[-1] == r"/":
        url += r"/"
    resp = requests.get(url + filename, stream=True)
    resp.raw.decode_content = True
    mem_fh = io.BytesIO(resp.raw.read())

    df = pd.read_feather(mem_fh)
    df.insert(1, "time_ms", df["match_time"])
    df.insert(2, "sub_id", df["data_group"])
    _mkh5_internal = [
        "data_group",
        "dblock_tick_idx",
        "dblock_ticks",
        "crw_ticks",
        "raw_evcodes",
        "epoch_match_tick_delta",
        "epoch_ticks",
        "idx",
        "dlim",
        "anchor_str",
        "match_str",
        "anchor_code",
        "match_code",
        "anchor_tick",
        "match_tick",
        "anchor_time_delta",
        "pygarv",
        "anchor_tick_delta",
        "is_anchor",
        "regexp",
        "match_time",
        "anchor_time",
    ]
    df.drop(columns=_mkh5_internal, inplace=True)
    return df
