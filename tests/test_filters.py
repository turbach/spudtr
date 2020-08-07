import pandas as pd
import numpy as np
import pytest
import spudtr.filters as filters
import matplotlib.pyplot as plt


xfve = pytest.mark.xfail(strict=True, reason=ValueError)

@pytest.mark.parametrize(
    "_allow_defaults",
    [True, pytest.param(False, marks=xfve)]
)
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
    ]
)
def test_check_filter_params_obligatory(_ftype, _cutoff, _srate, _allow_defaults):
    filt_params = filters.check_filter_params(
        ftype=_ftype,
        cutoff_hz=_cutoff,
        sfreq=_srate,
        allow_defaults=_allow_defaults
    )


@pytest.mark.parametrize("_width_hz", [5.0, None, pytest.param('_nn', marks=xfve)])
@pytest.mark.parametrize("_ripple_db", [53.0, None, pytest.param('_nn', marks=xfve)])
@pytest.mark.parametrize("_window", ["kaiser", "hamming", "hann", "blackman"])
def test_check_filter_params_optional(_width_hz, _ripple_db, _window):
    filt_params = filters.check_filter_params(
        ftype="lowpass",
        cutoff_hz=25.0,
        sfreq=25,
        width_hz=_width_hz,
        ripple_db=_ripple_db,
        window=_window,
        allow_defaults=True
    )


def test__suggest_epoch_length():
    sfreq = 250
    ripple_db = 60
    width_hz = 4
    N = filters._suggest_epoch_length(sfreq, ripple_db, width_hz)
    assert N == 230


@pytest.mark.parametrize(
    "_ftype,_cutoff_hz", [
        ("lowpass", 25.0),
        ("highpass", 10.0),
        ("bandpass", [10, 15]),
        ("bandstop", [10, 15])
    ]
)
@pytest.mark.parametrize(
    "_window", (None, "kaiser", "hamming", "hann", "blackman")
)
def test_show_filter(_ftype, _cutoff_hz, _window):
    cutoff_hz = 10.0
    sfreq = 250

    width_hz = 5.0
    ripple_db = 60.0
    filters.show_filter(
        ftype=_ftype,
        cutoff_hz=_cutoff_hz,
        sfreq=sfreq,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=_window,
    )

    filters.show_filter(ftype=_ftype, cutoff_hz=_cutoff_hz, sfreq=sfreq)
    filters.show_filter(ftype=_ftype, cutoff_hz=_cutoff_hz, sfreq=sfreq, show_output=False)
    plt.close('all')


@pytest.mark.parametrize(
    "_ftype,_cutoff_hz", [
        ("lowpass", 12.5),
        ("highpass", 20),
        ("bandpass", [10, 20]),
        ("bandstop", [5, 10]),
    ]
)
@pytest.mark.parametrize(
    "_window", ("kaiser", "hamming", "hann", "blackman")
)
def test_fir_filter_dt(_ftype, _cutoff_hz, _window):
    _params = filters.check_filter_params(
        ftype=_ftype,
        cutoff_hz=_cutoff_hz,
        window=_window,
        sfreq=250,
        allow_defaults=True
    )

    # create fakedata to show the filter
    freq_list = [10, 30]
    amplitude_list = [1.0, 1.0]
    t, y = filters._sins_test_data(freq_list, amplitude_list)
    testdata = pd.DataFrame({"fakedata": y})

    filt_test_df = filters.fir_filter_dt(testdata, ["fakedata"], **_params)
    assert isinstance(filt_test_df, pd.DataFrame)
    assert filt_test_df.columns.tolist() == ["fakedata"]

    with pytest.raises(TypeError) as excinfo:
        filt_test_df = filters.fir_filter_dt(y, ["fakedata"], **_params)
    assert "dt must be pandas.DataFrame or structured numpy.ndarray" in str(excinfo.value)


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


# @pytest.mark.parametrize(
#     "test_arg",
#     ["cutoff_hz", "width_hz", "ripple_db", "sfreq", "ftype", "window"],
# )
# def test_design_firwin_filter_args(test_arg):

#     # usable values
#     specs = dict(
#         cutoff_hz=20,
#         width_hz=5,
#         ripple_db=60,
#         sfreq=250,
#         ftype="lowpass",
#         window="kaiser",
#     )
#     specs[test_arg] = None
#     try:
#         taps = filters._design_firwin_filter(**specs)
#     except ValueError as fail:
#         assert str(fail) == f"{test_arg} is None, set a value"
#         pytest.xfail()


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

    _params = filters.check_filter_params(
        ftype="bandstop",
        window="hann",
        cutoff_hz=[18, 35],
        width_hz=5,
        ripple_db=60,
        sfreq= 250,
    )


    # build and apply the filter
    taps = filters._design_firwin_filter(**_params)
    filt_data = filters._apply_firwin_filter_data(y, taps)
    assert len(taps) == 183

    freq_list = [0.2, 3]
    amplitude_list = [1.0, 1.0]
    sampling_freq = 250
    t, y = filters._sins_test_data(freq_list, amplitude_list, sampling_freq)

    _params = filters.check_filter_params(
        ftype = "lowpass",
        cutoff_hz = 1,
        sfreq = sampling_freq,
        allow_defaults=True
    )
    with pytest.raises(ValueError) as excinfo:
        y_filt = filters.fir_filter_data(y, **_params)
    assert "filter I/O length mismatch" in str(excinfo.value)


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


@pytest.mark.parametrize(
    "_ftype,_cutoff_hz",
    [
        ["highpass", 20],
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
    # filters.filters_effect(_cutoff_hz, sfreq, ftype)

    # ftype = "lowpass"
    # cutoff_hz = 1
    # filters.filters_effect(cutoff_hz, sfreq, ftype)
    # ftype = "highpass"
    # cutoff_hz = 1
    # filters.filters_effect(cutoff_hz, sfreq, ftype)
    # ftype = "bandstop"
    # cutoff_hz = [1, 10]
    # filters.filters_effect(cutoff_hz, sfreq, ftype)
    # assert sfreq == 250
