"""utilities for epoched EEG data in a pandas.DataFrame """
import numpy as np
import pandas as pd
import bottleneck as bn
from scipy.signal import kaiserord, lfilter, firwin, freqz
from collections import defaultdict, OrderedDict


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


def _hdf_read_epochs(epochs_f, h5_group):
    """read tabular hdf5 epochs file, return as pd.DataFrame

    Parameter
    ---------
    epochs_f : str
        name of the recorded epochs file to load

    h5_group : str
        name of h5 group key

    Return
    ------
    df : pd.DataFrame
        columns in INDEX_NAMES are pd.MultiIndex axis 0
    """

    if h5_group is None:
        raise ValueError("You have to give h5_group key")
    else:
        epochs_df = pd.read_hdf(epochs_f, h5_group)

    _validate_epochs_df(epochs_df, epoch_id=None, time=None)
    return epochs_df


def _epochs_QC(epochs_df, eeg_streams, epoch_id=None, time=None):
    if epoch_id is None:
        epoch_id = "Epoch_idx"

    if time is None:
        time = "Time"

    # epochs_df must be a Pandas DataFrame.
    if not isinstance(epochs_df, pd.DataFrame):
        raise ValueError("epochs_df must be a Pandas DataFrame.")

    # eeg_streams must be a list of strings
    if not isinstance(eeg_streams, list) or not all(
        isinstance(item, str) for item in eeg_streams
    ):
        raise ValueError("eeg_streams should be a list of strings.")

    # all channels must be present as epochs_df columns
    missing_channels = set(eeg_streams) - set(epochs_df.columns)
    if missing_channels:
        raise ValueError(
            "eeg_streams should all be present in the epochs dataframe, "
            f"the following are missing: {missing_channels}"
        )

    # epoch_id and time must be the columns in the epochs_df
    _validate_epochs_df(epochs_df)

    # check no duplicate column names in index and regular columns
    names = list(epochs_df.index.names) + list(epochs_df.columns)
    deduped_names = list(OrderedDict.fromkeys(names))
    if deduped_names != names:
        raise ValueError("Duplicate column names not allowed.")

    # check values of epoch_id in every time group are the same, and unique in each time group
    # make our own copy so we are immune to modification to original table
    # epoch_id = "Epoch_idx"
    # time = "Time"
    table = epochs_df.copy().reset_index().set_index(epoch_id).sort_index()
    assert table.index.names == [epoch_id]

    snapshots = table.groupby([time])

    # check that snapshots across epochs have equal index by transitivity
    prev_group = None
    for idx, cur_group in snapshots:
        if prev_group is not None:
            if not prev_group.index.equals(cur_group.index):
                raise ValueError(
                    f"Snapshot {idx} differs from "
                    f"previous snapshot in {epoch_id} index:\n"
                    f"Current snapshot's indices:\n"
                    f"{cur_group.index}\n"
                    f"Previous snapshot's indices:\n"
                    f"{prev_group.index}"
                )
        prev_group = cur_group

    if not prev_group.index.is_unique:
        dupes = prev_group.index.filter(lambda x: len(x) > 1)
        raise ValueError(
            f"Duplicate values of epoch_id in each time group not allowed:\n{dupes}"
        )


def center_eeg(epochs_df, eeg_streams, start, stop):
    """center (a.k.a. "baseline") EEG amplitude on mean from start to stop

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

    # _validate_epochs_df(epochs_df)
    _epochs_QC(epochs_df, eeg_streams)

    #  times = epochs_df.index.unique("Time")

    times = epochs_df.Time.unique()  # Qin added
    if not start >= times[0]:
        start = times[0]
    if not stop <= times[-1]:
        stop = times[-1]
    assert start >= times[0]
    assert stop <= times[-1]

    # baseline subtraction ... compact expression, numpy is faster
    qstr = f"{start} <= Time and Time < {stop}"
    epochs_df_tmp = epochs_df.copy()
    epochs_df_tmp[eeg_streams] = epochs_df.groupby(["Epoch_idx"]).apply(
        lambda x: x[eeg_streams] - x.query(qstr)[eeg_streams].mean(axis=0)
    )

    # TO DO: for each epoch and each eeg stream, check that the mean amplitude
    # (start, stop) interval is 0 (to within rounding error).

    # check with numpy is close epochs_df[eeg_streams].query(qstr)

    # after center on, the mean inside interval should be zero

    after_mean = epochs_df_tmp.groupby(["Epoch_idx"]).apply(
        lambda x: x.query(qstr)[eeg_streams].mean(axis=0)
    )

    # a is afer_mean numpy array, and b is zero array same size as a

    a = after_mean.values
    b = np.zeros(after_mean.shape)

    # np.isclose(a,b)   #all false
    # np.isclose(a,b,atol=1e-05)  #most true, but some false

    # The absolute tolerance parameter: atol=1e-04
    TorF = np.isclose(a, b, atol=1e-04)
    if sum(sum(TorF)) == TorF.shape[0] * TorF.shape[1]:
        print("center_on is correct")
    else:
        raise ValueError(f"center_on is not successful with atol=1e-04")

    _validate_epochs_df(epochs_df_tmp)
    return epochs_df_tmp
