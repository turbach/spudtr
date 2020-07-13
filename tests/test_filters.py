import pandas as pd
import numpy as np
import pytest
import spudtr.filters as filters

# import matplotlib.pyplot as plt
# from scipy import signal, fftpack
# from pylab import *
# from scipy.signal import kaiserord, firwin, freqz, lfilter


def test__suggest_epoch_length():
    sfreq = 250
    ripple_db = 60
    width_hz = 4
    N = filters._suggest_epoch_length(sfreq, ripple_db, width_hz)
    assert N == 230


def test_show_filter():
    cutoff_hz = 10.0
    width_hz = 5.0
    ripple_db = 60.0
    sfreq = 250
    ftype = "lowpass"
    window = "hamming"
    filters.show_filter(cutoff_hz, width_hz, ripple_db, sfreq, ftype, window)
    assert sfreq == 250


def test_filter_show():
    cutoff_hz = 10.0
    sfreq = 250
    ftype = "lowpass"
    filters.filter_show(cutoff_hz, sfreq, ftype)
    window = "hamming"
    width_hz = 5.0
    ripple_db = 60.0
    filters.filter_show(
        cutoff_hz,
        sfreq,
        ftype,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )
    assert sfreq == 250


@pytest.mark.parametrize(
    "window_type", ("kaiser", "hamming", "hann", "blackman")
)
def test_fir_filter_dt(window_type):
    # create fakedata to show the filter
    freq_list = [10, 30]
    amplitude_list = [1.0, 1.0]
    t, y = filters._sins_test_data(freq_list, amplitude_list)
    testdata = pd.DataFrame({"fakedata": y})

    ftype = "lowpass"
    # window = "hamming"
    cutoff_hz = 12.5
    width_hz = 5
    ripple_db = 60
    sfreq = 250

    filt_test_df = filters.fir_filter_dt(
        testdata,
        ["fakedata"],
        ftype,
        window_type,
        cutoff_hz,
        width_hz,
        ripple_db,
        sfreq,
        trim_edges=False,
    )
    assert isinstance(filt_test_df, pd.DataFrame)
    assert filt_test_df.columns.tolist() == ["fakedata"]

    y_filt = filt_test_df["fakedata"]
    freq_list = [10]
    amplitude_list = [1.0]
    t, y1 = filters._sins_test_data(freq_list, amplitude_list)
    i1 = int(len(y1) / 2) - 20
    i2 = int(len(y1) / 2) + 20
    ya = y1[i1:i2]
    yb = y_filt[i1:i2]
    a = max(abs(ya - yb))
    TorF = bool(np.isclose(a, 0, atol=1e-01))
    assert TorF is True

    # Test for trim_edges=True
    testdata = pd.DataFrame({"Time": t, "fakedata": y})
    filt_test_df = filters.fir_filter_dt(
        testdata,
        ["fakedata"],
        ftype,
        window_type,
        cutoff_hz,
        width_hz,
        ripple_db,
        sfreq,
        trim_edges=True,
    )
    # check the trimming
    assert filt_test_df.shape == (193, 2)


def test_fir_filter_df():
    # create fakedata to show the filter
    freq_list = [10, 30]
    amplitude_list = [1.0, 1.0]
    t, y = filters._sins_test_data(freq_list, amplitude_list)
    testdata = pd.DataFrame({"fakedata": y})

    ftype = "lowpass"
    cutoff_hz = 12.5
    sfreq = 250

    filt_test_df = filters.fir_filter_df(
        testdata, ["fakedata"], cutoff_hz, sfreq, ftype
    )

    width_hz = 5
    ripple_db = 60
    window = "hamming"

    filt_test_df = filters.fir_filter_df(
        testdata,
        ["fakedata"],
        cutoff_hz,
        sfreq,
        ftype,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
        trim_edges=True,
    )

    assert isinstance(filt_test_df, pd.DataFrame)
    assert filt_test_df.columns.tolist() == ["fakedata"]
    # check the trimming
    # assert filt_test_df.shape == (193, 2)


@pytest.mark.parametrize(
    "_ndim, _ncol",
    [
        (1, 1),  # 1-D vector
        # (2, 1),  # 2-D table, 1 column
        # (2, 2),  # 2-D table, 2 columns
        pytest.param(
            3, 3, marks=pytest.mark.xfail(strict=True)
        ),  # 3-D should fail
    ],
)
@pytest.mark.parametrize("_trim_edges", [True, False])
def test_fir_filter_ndarray(_trim_edges, _ndim, _ncol):
    # test filtering on np.ndarray
    freq_list = [10]
    amplitude_list = [1.0]

    t, y = filters._sins_test_data(freq_list, amplitude_list)

    # propagate the test data to _ndim columns
    cols = np.ones(_ndim)
    dt = np.dtype([(f"fakedata_{i}", float) for i in range(_ndim)])

    if _ndim == 1:
        testdata = y.astype(dt)

    if _ndim > 1:
        # + cols does np.broadcast voodoo to replicate y
        testdata = (y.reshape(len(y), 1) + cols).astype(dt)

        if _ndim == 3:
            shape_2d = testdata.shape
            testdata = np.tile(testdata, _ncol).reshape(shape_2d + (_ndim,))

    ftype = "lowpass"
    window_type = "kaiser"
    cutoff_hz = 12.5
    width_hz = 5
    ripple_db = 60
    sfreq = 250

    filt_test_df = filters.fir_filter_dt(
        testdata,
        ["fakedata_0"],  # this always exists regardless of _ndim
        ftype,
        window_type,
        cutoff_hz,
        width_hz,
        ripple_db,
        sfreq,
        trim_edges=_trim_edges,
    )
    assert isinstance(filt_test_df, np.ndarray)
    assert filt_test_df.dtype.names == dt.names


@pytest.mark.xfail(strict=True)
def test_fir_filter_bad_obj():

    ftype, cutoff_hz, width_hz, ripple_db = "lowpass", 12.5, 5, 60
    window_type = "kaiser"
    sfreq = 250

    testdata = list(range(5))
    filters.fir_filter_dt(
        testdata,
        ["silly_rabbit_lists_dont_have_columns"],
        ftype,
        window_type,
        cutoff_hz,
        width_hz,
        ripple_db,
        sfreq,
        trim_edges=False,
    )


def test_mfreqz():
    cutoff_hz = 10.0
    width_hz = 5.0
    ripple_db = 60.0
    sfreq = 250
    ftype = "lowpass"
    window = "hamming"

    taps = filters._design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )
    fig1 = filters._mfreqz(taps, sfreq, cutoff_hz, width_hz, a=1)

    # bandstop
    ftype = "bandstop"
    window = "hann"
    cutoff_hz = [18, 35]
    width_hz = 5
    ripple_db = 60
    sfreq = 250
    taps2 = filters._design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )
    fig2 = filters._mfreqz(taps2, sfreq, cutoff_hz, width_hz, a=1)
    assert len(taps) == 183


def test_impz():
    cutoff_hz = 10.0
    width_hz = 5.0
    ripple_db = 60.0
    sfreq = 250
    ftype = "lowpass"
    window = "hamming"

    taps = filters._design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )
    fig = filters._impz(taps, a=1)
    assert len(taps) == 183


def test_design_firwin_filter():
    ftype = "highpass"
    window = "blackman"
    cutoff_hz = 20
    width_hz = 5
    ripple_db = 60
    sfreq = 250
    # build and apply the filter
    taps = filters._design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )
    assert len(taps) == 183

    # add another test for N is even.
    ftype = "highpass"
    window = "kaiser"
    width_hz = 4
    ripple_db = 60
    sfreq = 250
    # build and apply the filter
    taps2 = filters._design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )

    ftype = "bandpass"
    window = "kaiser"
    cutoff_hz = [22, 40]
    width_hz = 5
    ripple_db = 60
    sfreq = 250
    taps3 = filters._design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )
    window = "hann"
    taps4 = filters._design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )

    ftype = "bandstop"
    window = "kaiser"
    cutoff_hz = [18, 35]
    width_hz = 5
    ripple_db = 60
    sfreq = 250
    taps5 = filters._design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )


def test_apply_firwin_filter():
    # creat a fakedata to show the filter
    freq_list = [10, 25, 45]
    amplitude_list = [1.0, 1.0, 1.0]
    t, y = filters._sins_test_data(freq_list, amplitude_list)
    testdata = pd.DataFrame({"fakedata": y})

    ftype = "bandstop"
    window = "hann"
    cutoff_hz = [18, 35]
    width_hz = 5
    ripple_db = 60
    sfreq = 250

    epochs_df = testdata
    eeg_streams = ["fakedata"]
    # build and apply the filter
    taps = filters._design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )
    filt_epochs_df = filters._apply_firwin_filter(epochs_df, eeg_streams, taps)
    assert len(taps) == 183


@pytest.mark.parametrize("_show_plot", [True, False])
def test_sins_test_data(_show_plot):
    # creat a fakedata to show the filter
    freq_list = [10, 30]
    amplitude_list = [1.0, 1.0]
    t, y = filters._sins_test_data(
        freq_list, amplitude_list, show_plot=_show_plot
    )
    assert len(t) == 375


def test__trans_bwidth_ripple():
    ftype = "bandstop"
    window = "kaiser"
    cutoff_hz = [18, 35]
    sfreq = 250
    width_hz, ripple_db = filters._trans_bwidth_ripple(
        cutoff_hz, sfreq, ftype, window
    )
    assert ripple_db == 53

    ftype = "highpass"
    cutoff_hz = 12.5
    window = "hann"

    width_hz, ripple_db = filters._trans_bwidth_ripple(
        cutoff_hz, sfreq, ftype, window
    )
    assert ripple_db == 44

    window = "blackman"
    ftype = "bandstop"
    cutoff_hz = [18, 35]
    width_hz, ripple_db = filters._trans_bwidth_ripple(
        cutoff_hz, sfreq, ftype, window
    )
    assert ripple_db == 74
