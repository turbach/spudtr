import pandas as pd
import numpy as np
import pytest
import spudtr.filters as filters
import matplotlib.pyplot as plt


xfve = pytest.mark.xfail(strict=True, reason=ValueError)


@pytest.mark.parametrize("_allow_defaults", [True, pytest.param(False, marks=xfve)])
@pytest.mark.parametrize(
    "_ftype,_cutoff, _srate",
    [
        ("lowpass", 20.0, 250.0),
        ("highpass", 10.0, 250.0),
        ("bandpass", [15.0, 20.0], 250.0),
        ("bandstop", [20.0, 30.0], 250.0),
        pytest.param(None, 20.0, 250, marks=xfve),
        pytest.param("Xlowpass", 20.0, 250.0, marks=xfve),
        pytest.param("lowpass", None, 250, marks=xfve),
        pytest.param("lowpass", "_nn", 250, marks=xfve),
        pytest.param("lowpass", 20.0, None, marks=xfve),
        pytest.param("lowpass", 20.0, "_nn", marks=xfve),
    ],
)
def test_check_filter_params_obligatory(_ftype, _cutoff, _srate, _allow_defaults):
    filt_params = filters.check_filter_params(
        ftype=_ftype, cutoff_hz=_cutoff, sfreq=_srate, allow_defaults=_allow_defaults,
    )


@pytest.mark.parametrize("_width_hz", [5.0, None, pytest.param("_nn", marks=xfve)])
@pytest.mark.parametrize("_ripple_db", [53.0, None, pytest.param("_nn", marks=xfve)])
@pytest.mark.parametrize("_window", ["kaiser", "hamming", "hann", "blackman"])
def test_check_filter_params_defaults(_width_hz, _ripple_db, _window):
    filt_params = filters.check_filter_params(
        ftype="lowpass",
        cutoff_hz=25.0,
        sfreq=25,
        width_hz=_width_hz,
        ripple_db=_ripple_db,
        window=_window,
        allow_defaults=True,
    )


def test__suggest_epoch_length():
    sfreq = 250
    ripple_db = 60
    width_hz = 4
    N = filters._suggest_epoch_length(
        sfreq=sfreq, width_hz=width_hz, ripple_db=ripple_db
    )
    assert N == 230


@pytest.mark.parametrize(
    "_ftype,_cutoff_hz",
    [
        ("lowpass", 25.0),
        ("highpass", 10.0),
        ("bandpass", [10, 15]),
        ("bandstop", [10, 15]),
    ],
)
@pytest.mark.parametrize("_window", (None, "kaiser", "hamming", "hann", "blackman"))
def test_show_filter(_ftype, _cutoff_hz, _window):

    sfreq = 250
    width_hz = 5.0
    ripple_db = 60.0

    print("test", locals())
    filters.show_filter(
        ftype=_ftype,
        cutoff_hz=_cutoff_hz,
        sfreq=sfreq,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=_window,
    )

    filters.show_filter(ftype=_ftype, cutoff_hz=_cutoff_hz, sfreq=sfreq)
    filters.show_filter(
        ftype=_ftype, cutoff_hz=_cutoff_hz, sfreq=sfreq, show_output=False
    )
    plt.close("all")


@pytest.mark.parametrize(
    "_ftype,_cutoff_hz",
    [
        ("lowpass", 12.5),
        ("highpass", 20),
        ("bandpass", [10, 20]),
        ("bandstop", [5, 10]),
    ],
)
@pytest.mark.parametrize("_window", ("kaiser", "hamming", "hann", "blackman"))
def test_fir_filter_dt(_ftype, _cutoff_hz, _window):
    _params = filters.check_filter_params(
        ftype=_ftype,
        cutoff_hz=_cutoff_hz,
        window=_window,
        sfreq=250,
        allow_defaults=True,
    )

    # create data vector, dataframe, structured array,
    t, y = filters._sins_test_data([10], [1.0])  # freqs, amps
    test_df = pd.DataFrame({"fakedata": y})
    struct_arry = np.array(y, dtype=np.dtype([("fakedata", float)]))

    # dataframes and structured arrays should be OK
    for dt in [test_df, struct_arry]:
        filt_dt = filters.fir_filter_dt(dt, ["fakedata"], **_params)
        assert type(filt_dt) == type(dt)
        assert dt.shape == filt_dt.shape
        if isinstance(dt, pd.DataFrame):
            assert filt_dt.columns.tolist() == ["fakedata"]
        elif isinstance(dt, np.ndarray):
            assert filt_dt.dtype.names == ("fakedata",)

    # np.array should fail
    with pytest.raises(TypeError) as excinfo:
        filt_dt = filters.fir_filter_dt(y, ["fakedata"], **_params)
    assert "dt must be pandas.DataFrame or structured numpy.ndarray" in str(
        excinfo.value
    )


def test_mfreqz():
    ripple_db = 60.0
    sfreq = 250

    # low pass
    cutoff_hz = 10.0
    width_hz = 5.0
    ftype = "lowpass"
    window = "hamming"
    taps = filters._design_firwin_filter(
        ftype=ftype,
        cutoff_hz=cutoff_hz,
        sfreq=sfreq,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )
    _ = filters._mfreqz(
        b=taps, a=1, cutoff_hz=cutoff_hz, sfreq=sfreq, width_hz=width_hz
    )
    assert len(taps) == 183

    # bandstop
    ftype = "bandstop"
    window = "hann"
    cutoff_hz = [18, 35]
    width_hz = 5
    taps2 = filters._design_firwin_filter(
        ftype=ftype,
        cutoff_hz=cutoff_hz,
        sfreq=sfreq,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )
    _ = filters._mfreqz(
        b=taps2, a=1, cutoff_hz=cutoff_hz, sfreq=sfreq, width_hz=width_hz,
    )


def test_impz():
    cutoff_hz = 10.0
    width_hz = 5.0
    ripple_db = 60.0
    sfreq = 250
    ftype = "lowpass"
    window = "hamming"

    taps = filters._design_firwin_filter(
        ftype=ftype,
        sfreq=sfreq,
        cutoff_hz=cutoff_hz,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )
    fig = filters._impz(b=taps, a=1)
    assert len(taps) == 183


@pytest.mark.parametrize(
    "_ftype,_cutoff_hz,_width_hz,_window",
    [
        ("highpass", 20, 5, "blackman"),
        ("highpass", 20, 4, "kaiser"),
        ("bandpass", [22, 40], 5, "kaiser"),
        ("bandpass", [22, 40], 5, "hann"),
        ("bandstop", [18, 35], 5, "kaiser"),
    ],
)
def test_design_firwin_filter(_ftype, _cutoff_hz, _width_hz, _window):
    ripple_db = 60
    sfreq = 250
    taps = filters._design_firwin_filter(
        ftype=_ftype,
        cutoff_hz=_cutoff_hz,
        sfreq=sfreq,
        width_hz=_width_hz,
        ripple_db=ripple_db,
        window=_window,
    )


def test__apply_firwin_filter_data():
    # creat a fakedata to show the filter
    freq_list = [10, 25, 45]
    amplitude_list = [1.0, 1.0, 1.0]
    t, y = filters._sins_test_data(freq_list, amplitude_list)

    _params = filters.check_filter_params(
        ftype="bandstop",
        window="hann",
        cutoff_hz=[18, 35],
        width_hz=5,
        ripple_db=60,
        sfreq=250,
    )

    # build and apply the filter
    taps = filters._design_firwin_filter(**_params)
    filt_data = filters._apply_firwin_filter_data(y, taps)
    assert len(taps) == 183

    # test low pass fails when cutoff has two bounds
    freq_list = [0.2, 3]
    amplitude_list = [1.0, 1.0]
    sampling_freq = 250
    t, y = filters._sins_test_data(freq_list, amplitude_list, sampling_freq)

    _params = filters.check_filter_params(
        ftype="lowpass", cutoff_hz=1, sfreq=sampling_freq, allow_defaults=True
    )
    with pytest.raises(ValueError) as excinfo:
        y_filt = filters.fir_filter_data(y, **_params)
    assert "filter I/O length mismatch" in str(excinfo.value)


@pytest.mark.parametrize("window_type", ("kaiser", "hamming", "hann", "blackman"))
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
        cutoff_hz=cutoff_hz,
        sfreq=sfreq,
        ftype=ftype,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window_type,
    )

    # fetch
    _params = filters.check_filter_params(
        ftype=ftype, cutoff_hz=cutoff_hz, sfreq=sfreq, allow_defaults=True
    )
    filt_data = filters.fir_filter_data(data, **_params)


@pytest.mark.parametrize("_show_plot", [True, False])
def test_sins_test_data(_show_plot):
    # creat a fakedata to show the filter
    freq_list = [10, 30]
    amplitude_list = [1.0, 1.0]
    t, y = filters._sins_test_data(freq_list, amplitude_list, show_plot=_show_plot)
    assert len(t) == 375


@pytest.mark.parametrize(
    "_ft,_cthz,_win,_ripdb",
    [
        ("bandstop", [18, 35], "kaiser", 53),
        ("highpass", 12.5, "hann", 44),
        ("bandstop", [18, 35], "blackman", 74),
    ],
)
def test__trans_bwidth_ripple2(_ft, _cthz, _win, _ripdb):
    sfreq = 250
    width_hz, ripple_db = filters._trans_bwidth_ripple(
        ftype=_ft, cutoff_hz=_cthz, sfreq=sfreq, window=_win
    )
    assert ripple_db == _ripdb


@pytest.mark.parametrize(
    "_ftype,_cutoff_hz",
    [
        ["lowpass", 20],
        ["highpass", 20],
        ["bandpass", [22, 40]],
        ["bandstop", [18, 35]],
    ],
)
def test_filters_effect(_ftype, _cutoff_hz):

    width_hz = 5
    ripple_db = 60
    window = "blackman"
    sfreq = 250
    filters.filters_effect(
        ftype=_ftype,
        cutoff_hz=_cutoff_hz,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
        sfreq=sfreq,
    )
    plt.clf()
    plt.close("all")
