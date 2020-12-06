import numpy as np
import pandas as pd
import patsy
import mne
from mne.epochs import EpochsArray
from collections import OrderedDict

from spudtr.epf import _epochs_QC
from spudtr import RESOURCES_DIR

# RESOURCES_DIR

# default
EEG_LOCATIONS_F = RESOURCES_DIR / "eeg32.csv"


def _streams2mne_digmont(eeg_streams, eeg_locations_f=EEG_LOCATIONS_F):

    eeg_locs = pd.read_csv(eeg_locations_f)
    missing_streams = set(eeg_streams) - set(eeg_locs["stream"])
    if missing_streams:
        raise ValueError(f"eeg_streams not found in cap: {missing_streams}")

    df = eeg_locs.set_index("stream").loc[eeg_streams, :].reset_index()
    ch_names = df.stream.tolist()
    # pos = df[["x", "y", "z"]].values
    pos = 0.095 * df[["x", "y", "z"]].values
    dig_ch_pos = OrderedDict(zip(ch_names, pos))
    montage = mne.channels.make_dig_montage(ch_pos=dig_ch_pos, coord_frame="head")
    return montage


def categories2eventid(epochs_df, categories, epoch_id, time, time_stamp):
    """Build an MNE events array and event_id dict from one or more categorical variables.

    This uses patsy formulas and dummy coded (full rank) design
    matrixes to construct the MNE format event_id dictionary and
    corresponding events array (events x 3) for tagging and binning
    single-trial epochs for time-domain aggregation into
    ``mne.Evoked``, e.g., average event-related potentials (ERPs).

    A single category is split into the category levels, a.k.a conditions, bins,
    like so: ``~ 0 + a``.

    Multiple categories fully crossed like so: ``~ 0 +  a:b`` and ``~ 0 + a:b:c``
    
    Parameters
    ----------
    epochs_df : pandas.DataFrame
       A spudtr format epochs data with ``epoch_id``, ``time`` columns.

    categories : str or iterable of str
        The column name(s) of the categorical variables.

    epoch_id : str
        The name of the column with the unique epoch ids, e.g.,
        ``epoch_id``, ``Epoch_idx``.

    time : str
        The name of the column with the regular epoch time stamps, e.g., ``time``,
        ``time_ms``, ``time_s``.

    time_stamp : int The time stamp in the epoch to look up the
        categorical variable values, e.g., ``0``

    Returns
    -------
    mne_event_id : dict

       An MNE Python event_id dictionary where each item is ``label:
       event_code``.  The ``label`` is the column name from the patsy
       full-rank design matrix (incidence matrix) for the categories
       (thank you NJS). The ``event_code`` is the 1-based column index
       in the design matrix.

    mne_events : np.array, shape=(number_of_epochs, 3) there is one
       row for each epoch in ``epochs_df``. Each row is

         ``[epoch_id, 0, mne_event_code]`` 

       where ``mne_event_code`` is the newly
       constructed event code derived from the ``patsy`` design matrix
       column
        

    Examples
    --------
    Suppose at the specified time stamp the epochs_df categorical
    columns ``a`` and ``b`` have have the following levels: ``a: a1,
    a2``, ``b: b1, b2, b3``

    >>> categories2eventid(epochs_df, categories="a", epoch_id, time, time_stamp)
    event_ids = {
        "a[a1]": 1,
        "a[a2]": 2
    }


    >>> categories2eventid(epochs_df, categories="b", epoch_id, time, time_stamp)
    event_ids = {
        "b[b1]": 1,
        "b[b2]": 2,
        "b[b3]": 3
    }


    >>> categories2eventid(epochs_df, categories=["a", "b"], epoch_id, time, time_stamp)
    event_ids = {
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

    # check spudtr epochs format
    _ = _epochs_QC(epochs_df, categories, epoch_id=epoch_id, time=time)

    if time_stamp not in epochs_df[time].unique():
        raise ValueError(f"time_stamp {time_stamp} not found in epochs_df['{time}']")

    # slice the epoch row at the specified time_stamp, e.g., time==0
    # the category columns at this row are used to build the new
    # event_id dictionary
    events_df = epochs_df[epochs_df[time] == time_stamp].copy()
    for cat in categories:
        events_df[cat] = pd.Categorical(events_df[cat])

    # ensure dm is a full rank indicator matrix n columns = product of
    # factor levels w/ exactly one 1 in each row
    formula = "~ 0 + " + ":".join(categories)
    dm = patsy.dmatrix(formula, events_df)
    assert all(np.equal(1, [len(a) for a in map(lambda x: np.where(x == 1)[0], dm)]))
    dm_cols = dm.design_info.column_names

    # convert indidcator design  matrix to a 1-base vector that indexes
    # which column of dm has the indicator 1 via binary summation
    # e.g., dm = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] -> [1, 2, 3]

    dm_col_code = np.array(
        [np.where(dm[i, :] == 1)[0] + 1 for i in range(len(dm))]
    ).squeeze()
    assert dm_col_code.min() == 1
    assert dm_col_code.max() == dm.shape[1]

    # 1-base mne event code dict with column labels from patsy
    mne_event_id1 = dict([(dm_col, i + 1) for i, dm_col in enumerate(dm_cols)])

    # mne array: n-events x 3
    mne_events = np.stack(
        [events_df["epoch_id"].to_numpy(), np.zeros(len(events_df)), dm_col_code],
        axis=1,
    ).astype("int")
    # pdb.set_trace()
    real_event_id = np.unique(mne_events[:, 2])
    mne_event_id = {
        key: value for key, value in mne_event_id1.items() if value in real_event_id
    }
    return mne_event_id, mne_events


class EpochsSpudtr(EpochsArray):
    def __init__(
        self,
        input_fname,
        eeg_streams,
        categories,
        time_stamp,
        epoch_id=None,
        time=None,
        time_unit=None,
    ):

        epochs_df = pd.read_feather(input_fname)
        # check dataframe format
        _epochs_QC(epochs_df, eeg_streams, epoch_id=epoch_id, time=time)

        mne_event_ids, mne_events = categories2eventid(
            epochs_df, categories, epoch_id, time, time_stamp
        )

        # no point to an event ids dict without the actual events
        if mne_event_ids is not None and mne_events is None:
            raise ValueError("mne_events must also be specified to use mne_event_ids")

        # compute sfreq samples / second from the time-stamps. _epochs_QC should
        # ensure regular sampling interval but check anyway ...
        timestamps = epochs_df[time].unique()
        sampling_interval = list(set((timestamps - np.roll(timestamps, 1))[1:]))
        assert len(sampling_interval) == 1  # should be guaranteed by _epochs_QC
        sfreq = 1.0 / (sampling_interval[0] * time_unit)  # samples per second

        montage = _streams2mne_digmont(eeg_streams)
        info = mne.create_info(montage.ch_names, sfreq=sfreq, ch_types="eeg")
        info.set_montage(montage)  # for mne >0.19

        tmin = epochs_df[time].min() * time_unit
        epochs_data = []
        # import pdb; pdb.set_trace()
        for epoch_i in epochs_df[epoch_id].unique():
            epoch1 = epochs_df[montage.ch_names][
                epochs_df.epoch_id == epoch_i
            ].to_numpy()
            epochs_data.append(epoch1.T)
        super().__init__(
            epochs_data, info=info, tmin=tmin, events=mne_events, event_id=mne_event_ids
        )


# API
def read_spudtr_epochs(
    input_fname,
    eeg_streams,
    categories,
    time_stamp,
    epoch_id=None,
    time=None,
    time_unit=None,
):

    """Parameters
    ------------
    convert spudtr format epochs data to MNE Epochs
    input_fname : file name of spudtr format epochs data. 

    eeg_streams : list of str
        column names of the data streams

    categories : str or iterable of str
        The column name(s) of the categorical variables.

    time_stamp : int The time stamp in the epoch to look up the
        categorical variable values, e.g., ``0``

    epoch_id : str
        name of the epoch index

    time : str
        name of the time stamp index, e.g., "time_ms" 

    time_unit : float
        time stamp unit in seconds, e.g., 0.001 for milliseconds, 1.0
        for seconds

    Returns
    -------
    epochs : mne.Epochs

    """

    return EpochsSpudtr(
        input_fname, eeg_streams, categories, time_stamp, epoch_id, time, time_unit
    )
