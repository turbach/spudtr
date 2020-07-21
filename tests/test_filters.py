import pandas as pd
import numpy as np
import pytest
import spudtr.filters as filters


def test__suggest_epoch_length():
    sfreq = 250
    ripple_db = 60
    width_hz = 4
    N = filters._suggest_epoch_length(sfreq, ripple_db, width_hz)
    assert N == 230


def test_show_filter():
    cutoff_hz = 10.0
    sfreq = 250
    ftype = "lowpass"
    filters.show_filter(cutoff_hz, sfreq, ftype)
    window = "hamming"
    width_hz = 5.0
    ripple_db = 60.0
    filters.show_filter(
        cutoff_hz,
        sfreq,
        ftype,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )
    filters.show_filter(cutoff_hz, sfreq, ftype)
    filters.show_filter(cutoff_hz, sfreq, ftype, sample_effect=False)
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
    cutoff_hz = 12.5
    sfreq = 250

    filt_test_df = filters.fir_filter_dt(
        testdata, ["fakedata"], cutoff_hz, sfreq, ftype
    )

    width_hz = 5
    ripple_db = 60
    # window = "hamming"

    filt_test_df = filters.fir_filter_dt(
        testdata,
        ["fakedata"],
        cutoff_hz,
        sfreq,
        ftype,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window_type,
    )
    filt_test_df = filters.fir_filter_dt(
        testdata, ["fakedata"], cutoff_hz, sfreq, ftype
    )

    assert isinstance(filt_test_df, pd.DataFrame)
    assert filt_test_df.columns.tolist() == ["fakedata"]

    with pytest.raises(TypeError) as excinfo:
        filt_test_df = filters.fir_filter_dt(
            y, ["fakedata"], cutoff_hz, sfreq, ftype
        )
    assert "dt must be pandas.DataFrame or np.ndarray" in str(excinfo.value)


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


def test__apply_firwin_filter_data():
    # creat a fakedata to show the filter
    freq_list = [10, 25, 45]
    amplitude_list = [1.0, 1.0, 1.0]
    t, y = filters._sins_test_data(freq_list, amplitude_list)

    ftype = "bandstop"
    window = "hann"
    cutoff_hz = [18, 35]
    width_hz = 5
    ripple_db = 60
    sfreq = 250

    # build and apply the filter
    taps = filters._design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )
    filt_data = filters._apply_firwin_filter_data(y, taps)
    assert len(taps) == 183


@pytest.mark.parametrize(
    "window_type", ("kaiser", "hamming", "hann", "blackman")
)
def test_fir_filter_data(window_type):
    # creat a fakedata to show the filter
    freq_list = [10, 25, 45]
    amplitude_list = [1.0, 1.0, 1.0]
    t, data = filters._sins_test_data(freq_list, amplitude_list)

    ftype = "bandstop"
    window = "hann"
    cutoff_hz = [18, 35]
    width_hz = 5
    ripple_db = 60
    sfreq = 250

    filt_data = filters.fir_filter_data(
        data,
        cutoff_hz,
        sfreq,
        ftype,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window_type,
    )
    filt_data = filters.fir_filter_data(data, cutoff_hz, sfreq, ftype)
    assert sfreq == 250


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


def test_filters_effect():

    ftype = "highpass"
    window = "blackman"
    cutoff_hz = 20
    width_hz = 5
    ripple_db = 60
    sfreq = 250
    filters.filters_effect(
        cutoff_hz,
        sfreq,
        ftype,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )

    # add another test for N is even.
    ftype = "lowpass"
    window = "kaiser"
    width_hz = 4
    ripple_db = 60
    sfreq = 250
    filters.filters_effect(
        cutoff_hz,
        sfreq,
        ftype,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )

    ftype = "bandpass"
    window = "kaiser"
    cutoff_hz = [22, 40]
    width_hz = 5
    ripple_db = 60
    sfreq = 250
    filters.filters_effect(
        cutoff_hz,
        sfreq,
        ftype,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )

    ftype = "bandstop"
    window = "kaiser"
    cutoff_hz = [18, 35]
    width_hz = 5
    ripple_db = 60
    sfreq = 250
    filters.filters_effect(
        cutoff_hz,
        sfreq,
        ftype,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )
    filters.filters_effect(cutoff_hz, sfreq, ftype)
    assert sfreq == 250


"""
@pytest.mark.parametrize(
    "ftype,cutoff_hz",
    [
        ["highpass", 20],
        ["highpass", 20],
        ["bandpass", [22, 40]],
        ["bandstop", [18, 35]],
    ],
)
def test_filters_effect(ftype, cutoff_hz):

    # ftype = "highpass"
    window = "blackman"
    # cutoff_hz = 20
    width_hz = 5
    ripple_db = 60
    sfreq = 250
    filters.filters_effect(
        cutoff_hz,
        sfreq,
        ftype,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )
    filters.filters_effect(cutoff_hz, sfreq, ftype)
    assert sfreq == 250
"""
