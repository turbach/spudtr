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


# ------------------------------------------------------------
# ROUGH DEVELOPMENT BELOW HERE
# ------------------------------------------------------------

# TO DO: ADD LOGGING?

# expand the epoch interval so filter edge artifacts can be trimmed
# by halfwidth = (ntaps - 1) / 2, e.g.,
# filter half width = 35 samples * 4 ms / sample = 140 ms each edge
# RECORDED_EPOCH_START, RECORDED_EPOCH_STOP = -1640, 1644

# integer multiples of downsampled time-points, e.g., 4ms by 2 == 8ms
# DOWNSAMPLED_EPOCH_START, DOWNSAMPLED_EPOCH_STOP = -1496, 1496

# center on entire prestimulus interval ... don't include 0
# BASELINE_START, BASELINE_STOP = DOWNSAMPLED_EPOCH_START, -1


def get_logger(logger_name):
    """ build a standard output and file logger

    Parameters
    ----------
    logger_name : str

    """

    logr = logging.getLogger(logger_name)
    logr.setLevel(logging.DEBUG)

    log_sh = logging.StreamHandler()  # stream=sys.stdout)
    log_sh.setLevel(logging.DEBUG)

    log_fh = logging.FileHandler(logger_name + '.log', mode='w')
    log_fh.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter(
        "{name}:{levelname}:{message}", style='{'
    )
    log_fh.setFormatter(log_formatter)
    log_sh.setFormatter(log_formatter)
    logr.addHandler(log_fh)
    logr.addHandler(log_sh)

    return logr


def add_roi_columns(df, rois):
    """add a new "region of interest" column equal to the mean of existing
eeg columns

    Parameters
    ----------

    df : pd.DataFrame samples in rows, channels in column

    rois : dict
        each key is an new column roi label, each value is a list of
        column labels to pool

    Example
    -------

    rois = {
    'cp_roi': ['LMCe', 'RMCe', 'MiCe', 'MiPa', 'LDCe', 'RDCe']

    }

    """
    LOGGER.info(f'add_roi_columns {rois}')
    for roi, chans in rois.items():
        df[roi] = df[chans].mean(axis=1)  # average across chans columns
        # LOGGER.info('\nNew ROI head\n{0}'.format(df[chans + [roi]].head()))
        # LOGGER.info('\nNew ROI tail\n{0}'.format(df[chans + [roi]].tail()))
    return df


# ------------------------------------------------------------
# epochs processing function: all return modified copy
# ------------------------------------------------------------
def tag_peak_to_peak_excursions(epochs_df, eeg_streams, crit_ptp):

    # CRIT_PTP = 150.0
    ptp_codes = []

    LOGGER.info(
        f"""
    Tagging epochs with EEG peak-to-peak amplitude > {crit_ptp} microvolts
    """
    )

    # scan each epoch for peak-to-peak excursions
    for idx, epoch in epochs_df[eeg_streams].groupby('Epoch_idx'):
        ptps = epoch.to_numpy().ptp(axis=0).T  # pandas 24 T to keep it wide
        mask = (ptps > crit_ptp).astype('uint8')  # 0's and 1's each channel
        ptp_codes.append((idx, encode_eeg_fail(mask)))

    # before
    n_samps, n_epochs = check_epochs_shape(epochs_df)

    # propagate to all time points in the epoch
    ptp_excursion = np.repeat(ptp_codes, n_samps, axis=0)
    assert all(
        epochs_df.index.get_level_values('Epoch_idx') == ptp_excursion[:, 0]
    )
    epochs_df['ptp_excursion'] = ptp_excursion[:, 1]

    # after
    assert n_samps, n_epochs == check_epochs_shape(epochs_df)

    ptp_excursion_epoch_ids = [
        epoch_id for epoch_id, code in ptp_codes if code > 0
    ]
    msg = f"""
    ptp_excursions: {len(ptp_excursion_epoch_ids)}
    {ptp_excursion_epoch_ids}
    """
    LOGGER.info(msg)
    return epochs_df


def tag_flat_eeg(epochs_df, eeg_streams, blocking_min, blocking_n_samps):
    # streams: eeg data columns
    # blocking criteria, e.g., less than this microvolts in
    # in an interval of this many consecutive data points
    #   blocking_min = 0.01
    #   blocking_n_samps = 5

    LOGGER.info(
        f"""
    Tagging flat EEG epochs where peak-to-peak in any consecutive
    {blocking_n_samps} samples < {blocking_min} microvolts ...
    """
    )

    blocking_codes = []
    for idx, epoch in epochs_df[eeg_streams].groupby('Epoch_idx'):

        n_samps, n_epochs = check_epochs_shape(epochs_df)

        # numpy peak-to-peak
        epoch_arry = epoch.to_numpy()  # pandas 24
        win_mins = bn.move_min(epoch_arry, window=blocking_n_samps, axis=0)
        win_maxs = bn.move_max(epoch_arry, window=blocking_n_samps, axis=0)

        # minimum peak-to-peak of any window in the epoch
        win_ptp = np.nanmin(win_maxs - win_mins, axis=0)
        blck_mask = (win_ptp < blocking_min).astype('uint8')
        blocking_codes.append((idx, encode_eeg_fail(blck_mask)))

    blocked = np.repeat(blocking_codes, n_samps, axis=0)
    assert all(epochs_df.index.get_level_values('Epoch_idx') == blocked[:, 0])

    epochs_df['blocked'] = blocked[:, 1]
    assert n_samps, n_epochs == check_epochs_shape(epochs_df)

    blocked_epoch_ids = epochs_df.query("blocked != 0").index.unique(
        'Epoch_idx'
    )

    LOGGER.info(
        f"{len(blocked_epoch_ids)} blocked epochs: {blocked_epoch_ids}"
    )

    return epochs_df


def tag_garv_artifacts(epochs_df):

    # unpack and propagate garv codes into columns
    n_samps, n_epochs = check_epochs_shape(epochs_df)
    epochs_df['garv_blink'] = np.repeat(
        epochs_df.query("Time==0")['garv_reject']
        .apply(lambda x: 1 if x >= 48 else 0)
        .to_numpy(),  # pandas 24
        n_samps,
    )

    # propagate and rename garv rejects
    epochs_df['garv_screen'] = np.repeat(
        epochs_df.query("Time==0")['garv_reject']
        .apply(lambda x: "accept" if x == 0 else "reject")
        .to_numpy(),
        n_samps,
    )

    assert n_samps, n_epochs == check_epochs_shape(epochs_df)
    LOGGER.info(
        "Experimenter tagged artifacts in garv_reject,"
        " blinks in column 'garv_blink'"
    )

    return epochs_df


def consolidate_artifacts(epochs_df, eeg_screen_cols, eeg_screen_col):
    # combine eeg_screen_cols > 0 into one artifact indicator column
    n_samps, n_epochs = check_epochs_shape(epochs_df)
    epochs_df[eeg_screen_col] = np.repeat(
        epochs_df.query("Time==0")[eeg_screen_cols]
        .apply(
            lambda x: "accept" if x.sum() == 0 else "reject",
            1,  # axis 1 to iterate by row and sum across artifact columns
        )
        .to_numpy(),
        n_samps,
    )
    assert n_samps, n_epochs == check_epochs_shape(epochs_df)
    msg = f"""
    Artifact(s) in any of {eeg_screen_cols} are tagged as a reject
    in indicator column '{eeg_screen_col}'
    """
    LOGGER.info(msg)
    return epochs_df


def downsample_epochs(epochs_df, t0, t1, by):
    LOGGER.info(f"Downsampling ... decimating from {t0} to {t1} by {by}")

    # careful with index slice (start, stop, step) in pandas
    # start, stop are ms ROW LABELS, step is a ROW INDEX *COUNTER* not ms
    assert epochs_df.index.names == [
        'Epoch_idx',
        'expt',
        'sub_id',
        'item_id',
        'Time',
    ]

    time_slicer = pd.IndexSlice[:, :, :, :, slice(t0, t1, by)]
    epochs_df = epochs_df.loc[time_slicer, :].copy()
    return epochs_df


def drop_data_rejects(epochs_df, reject_column):
    """returns a copy of epochs_df where reject column is non-zero"""

    LOGGER.info(f"dropping rejects {reject_column}")

    # enforce all data in each epoch is marked all 0 good or all
    # (some kind of, possibly various) non-zero bad
    goods = []
    for epoch_idx, epoch_data in epochs_df.groupby("Epoch_idx"):
        if max(epoch_data[reject_column]) == 0:
            goods.append(epoch_idx)

    good_epochs_df = epochs_df.query("Epoch_idx in @goods").copy()
    good_epochs_df.sort_index(inplace=True)

    # sanity check the result
    for epoch_idx, epoch_data in good_epochs_df.groupby('Epoch_idx'):
        n_goods = len(np.where(epoch_data[reject_column] == 0)[0])
        if n_goods != len(epoch_data):
            raise Exception('uncaught exception')

    return good_epochs_df


def bimastoid_reference(epochs_df, eeg_streams, a2):
    """math-linked mastoid reference = subtract half of A2, all channels"""

    LOGGER.info(f"bimastoid_reference {a2}")

    half_A2 = epochs_df[a2].values / 2.0
    br_epochs_df = epochs_df.copy()
    for col in eeg_streams:
        br_epochs_df[col] = br_epochs_df[col] - half_A2

    return br_epochs_df


def center_on_interval(epochs_df, eeg_streams, start, stop):
    """eeg_stream subtract the mean of Time index slice(start:stop)

    Parameters
    ----------
    epochs_df : pd.DataFrame
        must have Epoch_idx and Time row index names

    eeg_streams: list of str
        column names to apply the transform

    start, stop : int,  start < stop
        basline interval Time values, stop is inclusive

    """

    msg = f"centering on interval {start} {stop}: {eeg_streams}"
    LOGGER.info(msg)

    validate_epochs_df(epochs_df)
    times = epochs_df.index.unique('Time')
    assert start >= times[0]
    assert stop <= times[-1]

    # baseline subtraction ... compact expression, numpy is faster
    qstr = f"{start} <= Time and Time < {stop}"
    epochs_df[eeg_streams] = epochs_df.groupby(["Epoch_idx"]).apply(
        lambda x: x[eeg_streams] - x.query(qstr)[eeg_streams].mean(axis=0)
    )
    validate_epochs_df(epochs_df)
    return epochs_df


def lowpass_filter_epochs(
    epochs_df, eeg_streams, cutoff_hz, width_hz, ripple_db, sfreq, trim_edges
):
    """FIRLS wrapper"""

    # ------------------------------------------------------------
    # encapsulate filter helpers
    # ------------------------------------------------------------
    def _get_firls_lp(cutoff_hz, width_hz, ripple_db, sfreq):
        """
        FIRLS at https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

        Parameters
        ----------

        cutoff_hz : float
            cutoff frequency in Hz, e.g., 5.0, 30.0

        width_hz : float
            transition band width start to stop in Hz

        ripple_db : float
            attenuation in the stop band, in dB, e.g., 24.0, 60.0

        sfreq : float
            sampling frequency, e.g., 250.0, 500.0

        """

        LOGGER.info(
            f"""
        Buildiing firls filter: cutoff_hz={cutoff_hz}, width_hz={width_hz}, ripple_db={ripple_db}, sfreq={sfreq}
        """
        )

        # Nyquist frequency
        nyq_rate = sfreq / 2.0

        # transition band width in normalizied frequency
        width = width_hz / nyq_rate

        # order and Kaiser parameter for the FIR filter.
        N, beta = kaiserord(ripple_db, width)

        # firwin with a Kaiser window to create a lowpass FIR filter.
        taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

        # frequency response ... useful for reporting
        w, h = freqz(taps)

        return taps, N, beta, w, h

    # ------------------------------------------------------------
    def _apply_firls_lp(df, columns, taps, N, beta):
        """apply the FIRLS filtering

        filtfilt() mangles data coming and going, doubles the order so
        instead we forward pass with lfilter() and compensate for the delay

        """

        assert len(taps) % 2 == 1  # enforce odd number of taps

        delay = int((len(taps) - 1) / 2)
        a = 1.0

        msg = f"""
        applying linear phase delay compensated filter.
        a: {a}, N: {N}, delay: {delay}
        taps:
        {taps}
        """
        LOGGER.info(msg)

        filt_df = df.copy()
        for column in columns:

            # forward pass
            filtered_column = lfilter(taps, a, df[column])

            # roll the phase shift by delay back to 0
            filt_df[column] = np.roll(filtered_column, -delay)

        return filt_df

    # build and apply the filter
    taps, N, beta, w, h = _get_firls_lp(
        cutoff_hz=cutoff_hz,
        width_hz=width_hz,
        ripple_db=ripple_db,
        sfreq=sfreq,
    )

    filt_epochs_df = _apply_firls_lp(epochs_df, eeg_streams, taps, N, beta)

    # optionally drop corrupted data
    if trim_edges:
        half_width = int(np.floor(N / 2))
        times = filt_epochs_df.index.unique('Time')
        start_good = times[
            half_width
        ]  # == first good sample b.c. 0-base index
        stop_good = times[-(half_width + 1)]  # last good sample, 0-base index
        return filt_epochs_df.query(
            "Time >= @start_good and Time <= @stop_good"
        )
    else:
        return filt_epochs_df

# ------------------------------------------------------------
# ROUGH DEVELOPMENT BELOW HERE
# ------------------------------------------------------------

# TO DO: ADD LOGGING?

# expand the epoch interval so filter edge artifacts can be trimmed
# by halfwidth = (ntaps - 1) / 2, e.g.,
# filter half width = 35 samples * 4 ms / sample = 140 ms each edge
# RECORDED_EPOCH_START, RECORDED_EPOCH_STOP = -1640, 1644

# integer multiples of downsampled time-points, e.g., 4ms by 2 == 8ms
# DOWNSAMPLED_EPOCH_START, DOWNSAMPLED_EPOCH_STOP = -1496, 1496

# center on entire prestimulus interval ... don't include 0
# BASELINE_START, BASELINE_STOP = DOWNSAMPLED_EPOCH_START, -1


def get_logger(logger_name):
    """ build a standard output and file logger

    Parameters
    ----------
    logger_name : str

    """

    logr = logging.getLogger(logger_name)
    logr.setLevel(logging.DEBUG)

    log_sh = logging.StreamHandler()  # stream=sys.stdout)
    log_sh.setLevel(logging.DEBUG)

    log_fh = logging.FileHandler(logger_name + '.log', mode='w')
    log_fh.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter(
        "{name}:{levelname}:{message}", style='{'
    )
    log_fh.setFormatter(log_formatter)
    log_sh.setFormatter(log_formatter)
    logr.addHandler(log_fh)
    logr.addHandler(log_sh)

    return logr


def add_roi_columns(df, rois):
    """add a new "region of interest" column equal to the mean of existing
eeg columns

    Parameters
    ----------

    df : pd.DataFrame samples in rows, channels in column

    rois : dict
        each key is an new column roi label, each value is a list of
        column labels to pool

    Example
    -------

    rois = {
    'cp_roi': ['LMCe', 'RMCe', 'MiCe', 'MiPa', 'LDCe', 'RDCe']

    }

    """
    LOGGER.info(f'add_roi_columns {rois}')
    for roi, chans in rois.items():
        df[roi] = df[chans].mean(axis=1)  # average across chans columns
        # LOGGER.info('\nNew ROI head\n{0}'.format(df[chans + [roi]].head()))
        # LOGGER.info('\nNew ROI tail\n{0}'.format(df[chans + [roi]].tail()))
    return df


# ------------------------------------------------------------
# epochs processing function: all return modified copy
# ------------------------------------------------------------
def tag_peak_to_peak_excursions(epochs_df, eeg_streams, crit_ptp):

    # CRIT_PTP = 150.0
    ptp_codes = []

    LOGGER.info(
        f"""
    Tagging epochs with EEG peak-to-peak amplitude > {crit_ptp} microvolts
    """
    )

    # scan each epoch for peak-to-peak excursions
    for idx, epoch in epochs_df[eeg_streams].groupby('Epoch_idx'):
        ptps = epoch.to_numpy().ptp(axis=0).T  # pandas 24 T to keep it wide
        mask = (ptps > crit_ptp).astype('uint8')  # 0's and 1's each channel
        ptp_codes.append((idx, encode_eeg_fail(mask)))

    # before
    n_samps, n_epochs = check_epochs_shape(epochs_df)

    # propagate to all time points in the epoch
    ptp_excursion = np.repeat(ptp_codes, n_samps, axis=0)
    assert all(
        epochs_df.index.get_level_values('Epoch_idx') == ptp_excursion[:, 0]
    )
    epochs_df['ptp_excursion'] = ptp_excursion[:, 1]

    # after
    assert n_samps, n_epochs == check_epochs_shape(epochs_df)

    ptp_excursion_epoch_ids = [
        epoch_id for epoch_id, code in ptp_codes if code > 0
    ]
    msg = f"""
    ptp_excursions: {len(ptp_excursion_epoch_ids)}
    {ptp_excursion_epoch_ids}
    """
    LOGGER.info(msg)
    return epochs_df


def tag_flat_eeg(epochs_df, eeg_streams, blocking_min, blocking_n_samps):
    # streams: eeg data columns
    # blocking criteria, e.g., less than this microvolts in
    # in an interval of this many consecutive data points
    #   blocking_min = 0.01
    #   blocking_n_samps = 5

    LOGGER.info(
        f"""
    Tagging flat EEG epochs where peak-to-peak in any consecutive
    {blocking_n_samps} samples < {blocking_min} microvolts ...
    """
    )

    blocking_codes = []
    for idx, epoch in epochs_df[eeg_streams].groupby('Epoch_idx'):

        n_samps, n_epochs = check_epochs_shape(epochs_df)

        # numpy peak-to-peak
        epoch_arry = epoch.to_numpy()  # pandas 24
        win_mins = bn.move_min(epoch_arry, window=blocking_n_samps, axis=0)
        win_maxs = bn.move_max(epoch_arry, window=blocking_n_samps, axis=0)

        # minimum peak-to-peak of any window in the epoch
        win_ptp = np.nanmin(win_maxs - win_mins, axis=0)
        blck_mask = (win_ptp < blocking_min).astype('uint8')
        blocking_codes.append((idx, encode_eeg_fail(blck_mask)))

    blocked = np.repeat(blocking_codes, n_samps, axis=0)
    assert all(epochs_df.index.get_level_values('Epoch_idx') == blocked[:, 0])

    epochs_df['blocked'] = blocked[:, 1]
    assert n_samps, n_epochs == check_epochs_shape(epochs_df)

    blocked_epoch_ids = epochs_df.query("blocked != 0").index.unique(
        'Epoch_idx'
    )

    LOGGER.info(
        f"{len(blocked_epoch_ids)} blocked epochs: {blocked_epoch_ids}"
    )

    return epochs_df


def tag_garv_artifacts(epochs_df):

    # unpack and propagate garv codes into columns
    n_samps, n_epochs = check_epochs_shape(epochs_df)
    epochs_df['garv_blink'] = np.repeat(
        epochs_df.query("Time==0")['garv_reject']
        .apply(lambda x: 1 if x >= 48 else 0)
        .to_numpy(),  # pandas 24
        n_samps,
    )

    # propagate and rename garv rejects
    epochs_df['garv_screen'] = np.repeat(
        epochs_df.query("Time==0")['garv_reject']
        .apply(lambda x: "accept" if x == 0 else "reject")
        .to_numpy(),
        n_samps,
    )

    assert n_samps, n_epochs == check_epochs_shape(epochs_df)
    LOGGER.info(
        "Experimenter tagged artifacts in garv_reject,"
        " blinks in column 'garv_blink'"
    )

    return epochs_df


def consolidate_artifacts(epochs_df, eeg_screen_cols, eeg_screen_col):
    # combine eeg_screen_cols > 0 into one artifact indicator column
    n_samps, n_epochs = check_epochs_shape(epochs_df)
    epochs_df[eeg_screen_col] = np.repeat(
        epochs_df.query("Time==0")[eeg_screen_cols]
        .apply(
            lambda x: "accept" if x.sum() == 0 else "reject",
            1,  # axis 1 to iterate by row and sum across artifact columns
        )
        .to_numpy(),
        n_samps,
    )
    assert n_samps, n_epochs == check_epochs_shape(epochs_df)
    msg = f"""
    Artifact(s) in any of {eeg_screen_cols} are tagged as a reject
    in indicator column '{eeg_screen_col}'
    """
    LOGGER.info(msg)
    return epochs_df


def downsample_epochs(epochs_df, t0, t1, by):
    LOGGER.info(f"Downsampling ... decimating from {t0} to {t1} by {by}")

    # careful with index slice (start, stop, step) in pandas
    # start, stop are ms ROW LABELS, step is a ROW INDEX *COUNTER* not ms
    assert epochs_df.index.names == [
        'Epoch_idx',
        'expt',
        'sub_id',
        'item_id',
        'Time',
    ]

    time_slicer = pd.IndexSlice[:, :, :, :, slice(t0, t1, by)]
    epochs_df = epochs_df.loc[time_slicer, :].copy()
    return epochs_df


def drop_data_rejects(epochs_df, reject_column):
    """returns a copy of epochs_df where reject column is non-zero"""

    LOGGER.info(f"dropping rejects {reject_column}")

    # enforce all data in each epoch is marked all 0 good or all
    # (some kind of, possibly various) non-zero bad
    goods = []
    for epoch_idx, epoch_data in epochs_df.groupby("Epoch_idx"):
        if max(epoch_data[reject_column]) == 0:
            goods.append(epoch_idx)

    good_epochs_df = epochs_df.query("Epoch_idx in @goods").copy()
    good_epochs_df.sort_index(inplace=True)

    # sanity check the result
    for epoch_idx, epoch_data in good_epochs_df.groupby('Epoch_idx'):
        n_goods = len(np.where(epoch_data[reject_column] == 0)[0])
        if n_goods != len(epoch_data):
            raise Exception('uncaught exception')

    return good_epochs_df


def bimastoid_reference(epochs_df, eeg_streams, a2):
    """math-linked mastoid reference = subtract half of A2, all channels"""

    LOGGER.info(f"bimastoid_reference {a2}")

    half_A2 = epochs_df[a2].values / 2.0
    br_epochs_df = epochs_df.copy()
    for col in eeg_streams:
        br_epochs_df[col] = br_epochs_df[col] - half_A2

    return br_epochs_df


def center_on_interval(epochs_df, eeg_streams, start, stop):
    """eeg_stream subtract the mean of Time index slice(start:stop)

    Parameters
    ----------
    epochs_df : pd.DataFrame
        must have Epoch_idx and Time row index names

    eeg_streams: list of str
        column names to apply the transform

    start, stop : int,  start < stop
        basline interval Time values, stop is inclusive

    """

    msg = f"centering on interval {start} {stop}: {eeg_streams}"
    LOGGER.info(msg)

    validate_epochs_df(epochs_df)
    times = epochs_df.index.unique('Time')
    assert start >= times[0]
    assert stop <= times[-1]

    # baseline subtraction ... compact expression, numpy is faster
    qstr = f"{start} <= Time and Time < {stop}"
    epochs_df[eeg_streams] = epochs_df.groupby(["Epoch_idx"]).apply(
        lambda x: x[eeg_streams] - x.query(qstr)[eeg_streams].mean(axis=0)
    )
    validate_epochs_df(epochs_df)
    return epochs_df


def lowpass_filter_epochs(
    epochs_df, eeg_streams, cutoff_hz, width_hz, ripple_db, sfreq, trim_edges
):
    """FIRLS wrapper"""

    # ------------------------------------------------------------
    # encapsulate filter helpers
    # ------------------------------------------------------------
    def _get_firls_lp(cutoff_hz, width_hz, ripple_db, sfreq):
        """
        FIRLS at https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

        Parameters
        ----------

        cutoff_hz : float
            cutoff frequency in Hz, e.g., 5.0, 30.0

        width_hz : float
            transition band width start to stop in Hz

        ripple_db : float
            attenuation in the stop band, in dB, e.g., 24.0, 60.0

        sfreq : float
            sampling frequency, e.g., 250.0, 500.0

        """

        LOGGER.info(
            f"""
        Buildiing firls filter: cutoff_hz={cutoff_hz}, width_hz={width_hz}, ripple_db={ripple_db}, sfreq={sfreq}
        """
        )

        # Nyquist frequency
        nyq_rate = sfreq / 2.0

        # transition band width in normalizied frequency
        width = width_hz / nyq_rate

        # order and Kaiser parameter for the FIR filter.
        N, beta = kaiserord(ripple_db, width)

        # firwin with a Kaiser window to create a lowpass FIR filter.
        taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

        # frequency response ... useful for reporting
        w, h = freqz(taps)

        return taps, N, beta, w, h

    # ------------------------------------------------------------
    def _apply_firls_lp(df, columns, taps, N, beta):
        """apply the FIRLS filtering

        filtfilt() mangles data coming and going, doubles the order so
        instead we forward pass with lfilter() and compensate for the delay

        """

        assert len(taps) % 2 == 1  # enforce odd number of taps

        delay = int((len(taps) - 1) / 2)
        a = 1.0

        msg = f"""
        applying linear phase delay compensated filter.
        a: {a}, N: {N}, delay: {delay}
        taps:
        {taps}
        """
        LOGGER.info(msg)

        filt_df = df.copy()
        for column in columns:

            # forward pass
            filtered_column = lfilter(taps, a, df[column])

            # roll the phase shift by delay back to 0
            filt_df[column] = np.roll(filtered_column, -delay)

        return filt_df

    # build and apply the filter
    taps, N, beta, w, h = _get_firls_lp(
        cutoff_hz=cutoff_hz,
        width_hz=width_hz,
        ripple_db=ripple_db,
        sfreq=sfreq,
    )

    filt_epochs_df = _apply_firls_lp(epochs_df, eeg_streams, taps, N, beta)

    # optionally drop corrupted data
    if trim_edges:
        half_width = int(np.floor(N / 2))
        times = filt_epochs_df.index.unique('Time')
        start_good = times[
            half_width
        ]  # == first good sample b.c. 0-base index
        stop_good = times[-(half_width + 1)]  # last good sample, 0-base index
        return filt_epochs_df.query(
            "Time >= @start_good and Time <= @stop_good"
        )
    else:
        return filt_epochs_df


# if __name__ == '__main__':

#     LOGGER.info('udck19 EEG single trial data wrangling')

#     # ------------------------------------------------------------
#     # 4. prepare recorded_epochs for modeling and measurement

#     #    Processing funtions and arguments for the epochs preparation, each a
#     #    tuple of (fnc, args, kwargs) is run in list order as
#     #
#     #    pr_epochs_df = fnc(pr_epochs_df, *args, **kwargs)
#     # ------------------------------------------------------------

#     PREP_STEPS = [

#         # lowpass antialising filter: 25 Hz for 5x oversampling after
#         # decimating by 2 to downsample from 250 to 125 samples/second
#         (
#             lowpass_filter_epochs,
#             (),
#             {
#                 "eeg_streams": ALL_EEG_STREAMS,
#                 "cutoff_hz": 25.0,
#                 "width_hz": 10.0,  # transition band
#                 "ripple_db": 48.0,  # pass band ripple
#                 "sfreq": 250.0,  # samples/second
#                 "trim_edges": True,  # discard 1st and last 1/2 filter width
#             }
#         ),

#         # downsample by 2 from 250 to 125 samples/second
#         (
#             downsample_epochs,
#             (),
#             {
#                 "t0": DOWNSAMPLED_EPOCH_START,
#                 "t1": DOWNSAMPLED_EPOCH_STOP,
#                 "by": 2
#             }
#         ),

#         # re-reference
#         (
#             bimastoid_reference,
#             (),
#             {
#                 "eeg_streams": ALL_EEG_STREAMS,
#                 "a2": "A2"
#             }
#         ),

#         # center each epoch on this interval
#         (
#             center_on_interval,
#             (),
#             {
#                 "eeg_streams": ALL_EEG_STREAMS,
#                 "start": BASELINE_START,
#                 "stop": BASELINE_STOP,
#             },
#         ),


#         # make the article item id for single trial analysis across
#         # expts *NOTE* Because of counterbalancing schemes, the
#         # article_item_id is only correct for *ARTICLES* and *MUST NOT
#         # BE USED FOR NOUN ITEM ANALYSIS*
#         (make_article_item_id, (), {}),

#         # tag peak-to-peak amplitude excursions
#         (
#             tag_peak_to_peak_excursions,
#             (),
#             {"eeg_streams": ALL_EEG_STREAMS, "crit_ptp": 150.0}
#         ),

#         # tag flat eeg
#         (
#             tag_flat_eeg,
#             (),
#             {
#                 "eeg_streams": ALL_EEG_STREAMS,
#                 "blocking_min": 0.01,
#                 "blocking_n_samps": 5,
#             }
#         ),

#         # unpack garv codes into epochs columns
#         (tag_garv_artifacts, (), {}),

#         # consolidate artifacts
#         (
#             consolidate_artifacts,
#             (),
#             {
#                 "eeg_screen_cols": ['garv_blink', 'ptp_excursion', 'blocked'],
#                 "eeg_screen_col": 'eeg_screen',
#             }
#         ),

#     ]
