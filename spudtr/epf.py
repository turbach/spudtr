"""utilities for epoched EEG data in a pandas.DataFrame """
import numpy as np
import pandas as pd
import bottleneck as bn
from scipy.signal import kaiserord, lfilter, firwin, freqz


def _validate_epochs_df(epochs_df, epoch_id=None, time=None):
    """check form and index of the epochs_df is as expected

    Parameters
    ----------
    epochs_df : pd.DataFrame

    """

    # LOGGER.info('validating epochs pd.DataFrame')
    if epoch_id is None:
        epoch_id = "Epoch_idx"

    if time is None:
        time = "Time"

    for key, val in {"epoch_id": epoch_id, "time": time}.items():
        # pdb.set_trace()
        if val not in epochs_df.columns:
            raise ValueError(f"{key} column {val} not found")


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


def center_eeg(epochs_df, eeg_streams, start, stop):
    """center (a.k.a. "baseline") EEG amplitude on mean amplitude from start to stop

    Parameters
    ----------
    epochs_df : pd.DataFrame
        must have Epoch_idx and Time row index names

    eeg_streams: list of str
        column names to apply the transform

    start, stop : int,  start < stop
        basline interval Time values, stop is inclusive

    """

    # msg = f"centering on interval {start} {stop}: {eeg_streams}"
    # LOGGER.info(msg)

    validate_epochs_df(epochs_df)
    times = epochs_df.index.unique("Time")
    assert start >= times[0]
    assert stop <= times[-1]

    # baseline subtraction ... compact expression, numpy is faster
    qstr = f"{start} <= Time and Time < {stop}"
    epochs_df[eeg_streams] = epochs_df.groupby(["Epoch_idx"]).apply(
        lambda x: x[eeg_streams] - x.query(qstr)[eeg_streams].mean(axis=0)
    )

    # TO DO: for each epoch and each eeg stream, check that the mean amplitude
    # (start, stop) interval is 0 (to within rounding error).

    validate_epochs_df(epochs_df)
    return epochs_df
