import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scipy import signal, fftpack
from pylab import *
import spudtr.filters as filters

import logging as LOGGER
from scipy.signal import kaiserord, firwin, freqz, lfilter


def test_suggest_epoch_length():
    sfreq = 250
    ripple_db = 60
    width_hz = 4
    N = filters.suggest_epoch_length(sfreq, ripple_db, width_hz)
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


@pytest.mark.parametrize(
    "window_type", ("kaiser", "hamming", "hann", "blackman")
)
def test_epochs_filters(window_type):
    # creat a fakedata to show the filter
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

    filt_test_df = filters.epochs_filters(
        testdata,
        ["fakedata"],
        ftype,
        # window = window_type,
        window_type,
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
    TorF = np.isclose(a, 0, atol=1e-01)

    # Test for trim_edges=True
    testdata = pd.DataFrame({"Time": t, "fakedata": y})
    filt_test_df = filters.epochs_filters(
        testdata,
        ["fakedata"],
        ftype,
        # window = window_type,
        window_type,
        cutoff_hz,
        width_hz,
        ripple_db,
        sfreq,
        trim_edges=True,
    )
    assert TorF == True


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
    assert len(taps) == 183


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


def test_sins_test_data():
    # creat a fakedata to show the filter
    freq_list = [10, 30]
    amplitude_list = [1.0, 1.0]
    t, y = filters._sins_test_data(freq_list, amplitude_list)
    assert len(t) == 375
