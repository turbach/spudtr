from pathlib import Path
import pytest
import spudtr.epf as epf
import spudtr.fake_epochs_data as fake_data
import spudtr.filters as filters
import numpy as np
import pandas as pd

import pdb

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "_f,h5_group",
    [
        ["sub000p3.epochs.h5", "p3"],
        ["sub000p5.epochs.h5", "p5"],
        ["sub000wr.epochs.h5", "wr"],
    ],
)
def test_hdf_read_epochs(_f, h5_group):
    epochs_df = epf._hdf_read_epochs(TEST_DATA_DIR / _f, h5_group)


# test by using one file
def test_epochs_QC():
    _f1, h5_group1 = "sub000wr.epochs.h5", "wr"
    epochs_df = epf._hdf_read_epochs(TEST_DATA_DIR / _f1, h5_group1)
    eeg_streams = ["MiPf", "MiCe", "MiPa", "MiOc"]
    epf._epochs_QC(epochs_df, eeg_streams)


def test_center_on():
    _f1, h5_group1 = "sub000wr.epochs.h5", "wr"
    epochs_df = epf._hdf_read_epochs(TEST_DATA_DIR / _f1, h5_group1)
    eeg_streams = ["MiPf", "MiCe", "MiPa", "MiOc"]
    start, stop = -50, 300
    epochs_df_centeron = epf.center_eeg(epochs_df, eeg_streams, start, stop)

    # after center on, the mean inside interval should be zero
    qstr = f"{start} <= Time and Time < {stop}"
    after_mean = epochs_df_centeron.groupby(["Epoch_idx"]).apply(
        lambda x: x.query(qstr)[eeg_streams].mean(axis=0)
    )

    # a is afer_mean numpy array, and b is zero array same size as a

    a = after_mean.values
    b = np.zeros(after_mean.shape)

    # np.isclose(a,b)   #all false
    # np.isclose(a,b,atol=1e-05)  #most true, but some false

    # The absolute tolerance parameter: atol=1e-04
    TorF = np.isclose(a, b, atol=1e-04)
    assert sum(sum(TorF)) == TorF.shape[0] * TorF.shape[1]


def test_drop_bad_epochs():
    _f1, h5_group1 = "sub000wr.epochs.h5", "wr"
    epochs_df = epf._hdf_read_epochs(TEST_DATA_DIR / _f1, h5_group1)
    epoch_id = "Epoch_idx"
    time = "Time"
    art_col = "log_flags"

    epochs_df_good = epf.drop_bad_epochs(epochs_df, art_col, epoch_id, time)
    epochs_df_good["new_col"] = 0

    # get the group of time == 0
    group = epochs_df.groupby([time]).get_group(0)
    good_idx = list(group[epoch_id][group[art_col] == 0])
    epochs_df_bad = epochs_df[~epochs_df[epoch_id].isin(good_idx)]
    assert epochs_df_good.shape[0] + epochs_df_bad.shape[0] == epochs_df.shape[0]


def test_re_reference():

    # create a fake data
    epochs_df = pd.DataFrame(
        np.array([[0, -3, 1, 2, 3], [0, -2, 4, 5, 6], [0, -1, 7, 8, 9]]),
        columns=["Epoch_idx", "Time", "a", "b", "c"],
    )

    eeg_streams = ["b", "c"]

    rs = ["a"]
    ref_type = "bimastoid"
    br_epochs_df = epf.re_reference(epochs_df, eeg_streams, rs, ref_type)
    assert list(br_epochs_df.b) == [1.5, 3.0, 4.5]
    rs = ["a"]
    ref_type = "new_common"
    br_epochs_df = epf.re_reference(epochs_df, eeg_streams, rs, ref_type)
    assert list(br_epochs_df.b) == [1, 1, 1]
    rs = ["a", "b"]
    ref_type = "common_average"
    br_epochs_df = epf.re_reference(epochs_df, eeg_streams, rs, ref_type)
    assert list(br_epochs_df.b) == [0.5, 0.5, 0.5]


@pytest.mark.parametrize(
    "eeg_streams,rs,ref_type,expected",
    [
        (["b", "c"], ["a"], "bimastoid", [1.5, 3.0, 4.5]),
        (["b", "c"], ["a"], "new_common", [1, 1, 1]),
        (["b", "c"], ["a", "b"], "common_average", [0.5, 0.5, 0.5]),
        pytest.param(
            ["b", "c"],
            ["a", "b"],
            "Xcommon_average",
            [0.5, 0.5, 0.5],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_re_reference_2(eeg_streams, rs, ref_type, expected):

    # create a fake data
    epochs_df = pd.DataFrame(
        np.array([[0, -3, 1, 2, 3], [0, -2, 4, 5, 6], [0, -1, 7, 8, 9]]),
        columns=["Epoch_idx", "Time", "a", "b", "c"],
    )

    br_epochs_df = epf.re_reference(epochs_df, eeg_streams, rs, ref_type)

    assert list(br_epochs_df.b) == expected


def test_lowpass_filter():

    # creat a fakedata to show the filter
    freq_list = [10, 30]
    amplitude_list = [1.0, 1.0]
    t, y = filters._sins_test_data(freq_list, amplitude_list)
    testdata = pd.DataFrame({"fakedata": y})

    ftype = "lowpass"
    window = "kaiser"
    cutoff_hz = 12.5
    width_hz = 5
    ripple_db = 60
    sfreq = 250

    filt_test_df = filters.epochs_filters(
        testdata,
        ["fakedata"],
        ftype,
        window,
        cutoff_hz,
        width_hz,
        ripple_db,
        sfreq,
        trim_edges=False,
    )
    y_filt = filt_test_df["fakedata"]
    freq_list = [10]
    amplitude_list = [1.0]
    t, y1 = filters._sins_test_data(freq_list, amplitude_list)
    i1 = int(len(y1) / 2) - 20
    i2 = int(len(y1) / 2) + 20
    ya = y1[i1:i2]
    yb = y_filt[i1:i2]
    a = max(abs(ya - yb))
    TorF = np.isclose(a, 0, atol=1e-02)
    assert TorF == True


def test_bandpass_filter():

    # creat a fakedata to show the filter
    freq_list = [10, 30, 60]
    amplitude_list = [1.0, 1.0, 1.0]
    t, y = filters._sins_test_data(freq_list, amplitude_list)
    testdata = pd.DataFrame({"fakedata": y})

    ftype = "bandpass"
    window = "kaiser"
    cutoff_hz = [22, 40]
    width_hz = 5
    ripple_db = 60
    sfreq = 250

    filt_test_df = filters.epochs_filters(
        testdata,
        ["fakedata"],
        ftype,
        window,
        cutoff_hz,
        width_hz,
        ripple_db,
        sfreq,
        trim_edges=False,
    )
    y_filt = filt_test_df["fakedata"]

    freq_list = [30]
    amplitude_list = [1.0]
    t, y1 = filters._sins_test_data(freq_list, amplitude_list)

    i1 = int(len(y1) / 2) - 20
    i2 = int(len(y1) / 2) + 20
    ya = y1[i1:i2]
    yb = y_filt[i1:i2]
    a = max(abs(ya - yb))
    TorF = np.isclose(a, 0, atol=1e-03)
    assert TorF == True
