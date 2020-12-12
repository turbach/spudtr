from pathlib import Path
import re

# single source the python package version
__version__ = "0.0.14"

DATA_DIR = Path(__file__).parents[0] / "data"
RESOURCES_DIR = Path(__file__).parents[0] / "resources"

# DEPRECATED in v.0.0.9, to be removed v0.0.11
# local EEG files for testing and docs in DATA_DIR
P3_F = "gh_sub000p3.epochs.h5"
P5_F = "gh_sub000p5.epochs.h5"
WR_F = "gh_sub000wr.epochs.h5"


# Zenodo epochs files for testing and docs
DATA_URL = r"https://zenodo.org/record/3968485/files/"
P3_100_FEATHER = "sub000p3.ms100.epochs.feather"
P5_100_FEATHER = "sub000p50.ms100.epochs.feather"
WR_100_FEATHER = "sub000wr.ms100.epochs.feather"
PM_100_FEATHER = "sub000pm.ms100.epochs.feather"

P3_1500_FEATHER = "sub000p3.ms1500.epochs.feather"
P5_1500_FEATHER = "sub000p50.ms1500.epochs.feather"
WR_1500_FEATHER = "sub000wr.ms1500.epochs.feather"
PM_1500_FEATHER = "sub000pm.ms1500.epochs.feather"


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
    """fetch and cache feather format demo EEG epochs data 

    default = Zenodo eeg-workshops/mkpy_data_examples/data, v0.0.3
              https://doi.org/10.5281/zenodo.3968485/files

    Parameters
    ----------
    filename : str
       file to fetch

    url : str {DATA_URL}
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

    # shortcut if previously downloaded
    if (DATA_DIR / filename).exists():
        return pd.read_feather(DATA_DIR / filename)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    # otherwise download
    print(f"downloading ./spudtr/data/{filename} from {url} ... please wait")
    if not url[-1] == r"/":
        url += r"/"
    resp = requests.get(url + filename, stream=True)
    resp.raw.decode_content = True
    mem_fh = io.BytesIO(resp.raw.read())
    df = pd.read_feather(mem_fh)

    df["epoch_id"] = df["epoch_id"].astype(int)
    df.insert(1, "time_ms", df["match_time"])
    df.insert(2, "sub_id", df["data_group"])
    df.insert(3, "eeg_artifact", df["log_flags"])
    _mkh5_internal = [
        "data_group",
        "dblock_tick_idx",
        "dblock_ticks",
        "crw_ticks",
        "raw_evcodes",
        "log_flags",
        "epoch_match_tick_delta",
        "epoch_ticks",
        "idx",
        "dlim",
        "match_str",
        "match_code",
        "match_tick",
        "match_time",
        "match_group",
        "is_anchor",
        "anchor_str",
        "anchor_code",
        "anchor_tick",
        "anchor_tick_delta",
        "anchor_time",
        "anchor_time_delta",
        "regexp",
        "pygarv",
    ]
    df.drop(columns=_mkh5_internal, inplace=True)
    df.to_feather(DATA_DIR / filename)
    return df
