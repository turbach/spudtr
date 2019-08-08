"""utilities for epoched EEG data in pandas.DataFrame with columns:
`epoch_idx`, `time`, `ch_01`, `ch_02`, ..., `ch_J`"

"""
import numpy as np
import pandas as pd
import bottleneck as bn
from scipy.signal import kaiserord, lfilter, firwin, freqz

import pdb


def _validate_epochs_df(epochs_df, epoch_id=None, time=None):
    """check form and index of the epochs_df is as expected

    Parameters
    ----------
    epochs_df : pd.DataFrame

    """

    # LOGGER.info('validating epochs pd.DataFrame')
    if epoch_id is None:
        epoch_id = 'Epoch_idx'

    if time is None:
        time = 'Time'

    for key, val in {'epoch_id': epoch_id, 'time': time}.items():
        # pdb.set_trace()
        if val not in epochs_df.columns:
            raise ValueError(f'{key} column {val} not found')


def _hdf_read_epochs(epochs_f, h5_group=None):
    """read tabular hdf5 epochs file, return as pd.DataFrame

    Parameter
    ---------
    epochs_f : str
        name of the recorded epochs file to load

    Return
    ------
    df : pd.DataFrame
        columns in INDEX_NAMES are pd.MultiIndex axis 0
    """

    if h5_group is None:
        epochs_df = pd.read_hdf(epochs_f)
    else:
        epochs_df = pd.read_hdf(epochs_f, h5_group)

    _validate_epochs_df(epochs_df, epoch_id=None, time=None)
    return epochs_df
