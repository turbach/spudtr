import numpy as np
import pandas as pd
import patsy
import mne
from collections import OrderedDict

from spudtr.epf import _epochs_QC


def streams2mne_digmont(eeg_streams):

    cap26 = pd.read_csv(
        "/home/qiz001/zq/Projects/spudtr/spudtr/data/cap26.csv"
    )
    missing_streams = set(eeg_streams) - set(cap26["stream"])
    if missing_streams:
        raise ValueError(f"eeg_streams not found in cap: {missing_streams}")

    df = cap26.set_index("stream").loc[eeg_streams, :].reset_index()
    ch_names = df.stream.to_list()
    pos = df[["x", "y", "z"]].values
    dig_ch_pos = OrderedDict(zip(ch_names, pos))
    montage = mne.channels.make_dig_montage(
        ch_pos=dig_ch_pos, coord_frame="head"
    )
    return montage


def _categories2eventid(epochs_df, categories, epoch_id, time, time_stamp):
    """build mne events and event_id with patsy 


    This uses patsy formulas and full rank dummy coded design matrixes to build
    an mne format event_id dictionaries and corresponding events array (events x 3).

    A single category is split into the category levels, a.k.a conditions, bins,
    like so "~ 0 + a".

    Multiple categories fully crossed like so: "~ 0 +  a:b" and "~ 0 + a:b:c"
    

    Parameters
    ----------
    categories : str or list of str
        column name(s) with string or pd.Categorical values 

    epochs_df : pandas.DataFrame
       spudtr format epoch_id, time data



    Example
    -------

    For categorical columns: levels

        a: a1, a2
        b: b1, b2, b3


    If categories="a"  then

        epoch_id = {
            "a[a1]": 1,
            "a[a2]": 2
        }

    If categories="b"  then

        epoch_id = {
            "b[b1]": 1,
            "b[b2]": 2,
            "b[b3]": 3
    }


    If categories=["a", "b"] then

        epoch_id = {
            'a[a1]:b[b1]': 1,
            'a[a2]:b[b1]': 2,
            'a[a1]:b[b2]': 3,
            'a[a2]:b[b2]': 4,
            'a[a1]:b[b3]': 5,
            'a[a2]:b[b3]': 6
        }

    """

    # modicum of guarding
    if isinstance(categories, str):
        categories = [categories]

    # _epochs_QC(epochs_df, data_streams, epoch_id=EPOCH_ID, time=TIME)
    _ = _epochs_QC(epochs_df, categories, epoch_id=epoch_id, time=time)

    if time_stamp not in epochs_df[time].unique():
        raise ValueErrorf(
            f"time_stamp {time_stamp} not found in epochs_df['{time}']"
        )

    # row slicing preserves the original row index which is handy for events array
    events_df = epochs_df[epochs_df[time] == time_stamp]

    # ensure dm is a full rank indicator matrix n columns = product of
    # factor levels w/ exactly one 1 in each row
    formula = "~ 0 + " + ":".join(categories)
    dm = patsy.dmatrix(formula, events_df)
    assert all(
        np.equal(1, [len(a) for a in map(lambda x: np.where(x == 1)[0], dm)])
    )
    dm_cols = dm.design_info.column_names

    # convert dm indidcator matrix to a 1-base vector that indexes
    # which column of dm has the 1 via binary summation
    # e.g., dm = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] -> [1, 2, 3]
    dm_col_code = (
        np.log2(dm.dot(1 << np.arange(dm.shape[-1]))).astype("int") + 1
    )
    assert dm_col_code.min() == 1
    assert dm_col_code.max() == dm.shape[1]

    # 1-base mne event code dict with column labels from patsy
    mne_event_id = dict([(dm_col, i + 1) for i, dm_col in enumerate(dm_cols)])

    # mne array: n-events x 3
    mne_events = np.stack(
        [events_df.index.to_numpy(), np.zeros(len(events_df)), dm_col_code],
        axis=1,
    ).astype("int")

    return mne_event_id, mne_events


def spudtr2mne(epochs_df, eeg_streams, time, epoch_id, sfreq):

    montage = streams2mne_digmont(eeg_streams)

    # create mne epochs from EpochsArray and show them
    info = mne.create_info(
        montage.ch_names, sfreq=sfreq, ch_types="eeg", montage=montage
    )

    epochs_data = []
    n_epochs_begin = 0
    n_epochs_end = max(epochs_df[epoch_id])
    # n_epochs = n_epochs_end - n_epochs_begin

    # import pdb; pdb.set_trace()
    for epoch_i in range(n_epochs_begin, n_epochs_end):
        epoch1 = epochs_df[info["ch_names"]][
            epochs_df.epoch_id == epoch_i
        ].to_numpy()
        epochs_data.append(epoch1.T)
    epochs = mne.EpochsArray(epochs_data, info=info)
    return epochs
