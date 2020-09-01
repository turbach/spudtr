"""FIR filter wrappers and utility functions. 

All filter functions require explicit parameters except
``check_filter_params()`` and ``show_filter()`` which will,
respectively, return and display suggested defaults for ``window``,
``width_hz`` (transition band) and ``ripple_db`` if these are not
specified.

"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy import signal, fftpack

import logging as LOGGER
from scipy.signal import kaiserord, firwin, freqz, lfilter


FTYPES = ["lowpass", "highpass", "bandpass", "bandstop"]
WINDOWS = ["kaiser", "hamming", "hann", "blackman"]


# ------------------------------------------------------------
# "private"-ish functions


def _trans_bwidth_ripple(ftype=None, cutoff_hz=None, sfreq=None, window=None):

    """
    Calculate reasonable default transition width and ripple dB

    Parameters
    ----------

    ftype : string
        filter type, e.g., 'lowpass' , 'highpass', 'bandpass', 'bandstop'
    cutoff_hz : float or 1D array_like
        cutoff frequency in Hz
    sfreq : float
        sampling frequency per second, e.g., 250.0, 500.0
    window : string
        window type for firwin, e.g., 'kaiser','hamming','hann','blackman'

    Returns
    -------
    width_hz : float
        transition band width start to stop in Hz
    ripple_db : float
        attenuation in the stop band, in dB
    """
    for kwarg in [ftype, cutoff_hz, sfreq, window]:
        assert kwarg is not None

    if ftype.lower() == "lowpass":
        width_hz = min(max(cutoff_hz * 0.25, 2), cutoff_hz)
    elif ftype.lower() == "highpass":
        width_hz = min(max(cutoff_hz * 0.25, 2.0), sfreq / 2.0 - cutoff_hz)
    elif ftype.lower() == "bandpass" or ftype.lower() == "bandstop":
        l = min(max(cutoff_hz[0] * 0.25, 2), cutoff_hz[0])
        h = min(max(cutoff_hz[1] * 0.25, 2.0), sfreq / 2.0 - cutoff_hz[1])
        width_hz = (l + h) / 2

    if window.lower() == "kaiser" or window.lower() == "hamming":
        ripple_db = 53
    elif window.lower() == "hann":
        ripple_db = 44
    elif window.lower() == "blackman":
        ripple_db = 74

    return width_hz, ripple_db


def _suggest_epoch_length(sfreq=None, width_hz=None, ripple_db=None):

    """
    Parameters
    ----------
    sfreq : float
        sampling frequency, i.e. 250.0
    width_hz : float
        width of transition region in Hz
    ripple_db : float
        ripple in dB

    Examples
    --------
    >>> sfreq = 250
    >>> ripple_db = 60
    >>> width_hz = 4
    >>> suggest_epoch_length(sfreq, ripple_db, width_hz)
    your epoch length should be  230  points, or  0.92  seconds at least.
    """

    for kwarg in [sfreq, width_hz, ripple_db]:
        assert kwarg is not None

    # Nyquist frequency
    nyq_rate = sfreq / 2.0

    # transition band width in normalizied frequency
    width = width_hz / nyq_rate

    # order and Kaiser parameter for the FIR filter.
    # The parameters returned by this function are generally used for the window method with firwin.
    N, beta = kaiserord(ripple_db, width)

    N = N + 2

    print(
        "your epoch length should be ",
        N,
        " points, or ",
        N / sfreq,
        " seconds at least. ",
    )
    return N


def _mfreqz(b=None, a=1, cutoff_hz=None, sfreq=None, width_hz=None):

    """ Plot the frequency and phase response of a digital filter.

    Parameters
    ----------
    b : array_like
        numerator of a linear filter
    a : array_like
        denominator of a linear filter
    cutoff_hz : float or 1D array_like
        cutoff frequency in Hz
    sfreq : float
        sampling frequency, e.g., 250.0, 500.0
    width_hz : float
        transition band width start to stop in Hz

    Returns
    -------
    fig : `~.figure.Figure`
    """

    for kwarg in [b, cutoff_hz, sfreq, width_hz, a]:
        assert kwarg is not None

    w, h = signal.freqz(b, a)
    h_dB = 20 * np.log10(abs(h))

    fig, (ax_freq, ax_freq_phase) = plt.subplots(2, 1)
    # make a little extra space between the subplots
    fig.subplots_adjust(hspace=0.6)

    # frequency response plot
    nyq_rate = sfreq / 2.0
    ax_freq.plot((w / np.pi) * nyq_rate, abs(h))
    cutoff_hz = np.atleast_1d(cutoff_hz)
    lstyle = {"linestyle": "--", "lw": 1, "color": "r"}
    if cutoff_hz.size == 1:
        ax_freq.axvline(cutoff_hz + width_hz / 2, **lstyle)
        ax_freq.axvline(cutoff_hz - width_hz / 2, **lstyle)
    else:
        ax_freq.axvline(cutoff_hz[0] + width_hz / 2, **lstyle)
        ax_freq.axvline(cutoff_hz[0] - width_hz / 2, **lstyle)
        ax_freq.axvline(cutoff_hz[1] + width_hz / 2, **lstyle)
        ax_freq.axvline(cutoff_hz[1] - width_hz / 2, **lstyle)

    ax_freq.set_ylabel("Gain")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_title(r"Frequency Response")

    # ax_freq.set_xlim(0, nyq_rate)
    ax_freq.set_xlim(-10, nyq_rate / 2)
    ax_freq.grid(linestyle="--")

    # frequency-phase plot
    ax_freq_phase.plot(w / max(w), h_dB, "b")
    ax_freq_phase.set_ylim(-150, 5)
    ax_freq_phase.set_ylabel("Magnitude (db)", color="b")
    ax_freq_phase.set_xlabel(r"Normalized Frequency (x$\pi$rad/sample)")
    ax_freq_phase.set_title(r"Frequency and Phase response")
    ax_freq_phaseb = ax_freq_phase.twinx()
    h_Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
    ax_freq_phaseb.plot(w / max(w), h_Phase, "g")
    ax_freq_phaseb.set_ylabel("Phase (radians)", color="g")
    ax_freq_phase.grid(linestyle="--")

    return fig


def _impz(b=None, a=1):

    """ Plot step and impulse response.

    Parameters
    ----------
    b : array_like
        numerator of a linear filter
    a : array_like
        denominator of a linear filter

    Returns
    -------
    fig : `~.figure.Figure`
    """

    for kwarg in [b, a]:
        assert kwarg is not None

    l = len(b)
    impulse = np.repeat(0.0, l)
    impulse[0] = 1.0
    x = np.arange(0, l)
    response = signal.lfilter(b, a, impulse)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # make a little extra space between the subplots
    fig.subplots_adjust(hspace=0.6)

    ax1.stem(x, response, use_line_collection=True)
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel(r"n (samples)")
    ax1.set_title(r"Impulse response")

    step = np.cumsum(response)
    ax2.stem(x, step, use_line_collection=True)
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel(r"n (samples)")
    ax2.set_title(r"Step response")
    return fig


def _design_firwin_filter(
    ftype=None, cutoff_hz=None, sfreq=None, width_hz=None, ripple_db=None, window=None
):
    """calculate odd length, symmetric, linear phase FIR filter coefficients

    FIRLS at https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

    Parameters
    ----------

    ftype : string
        filter type, one of 'lowpass' , 'highpass', 'bandpass', 'bandstop'

    cutoff_hz : float or 1D array_like
        cutoff frequency in Hz, e.g., 5.0, 30.0 for lowpass or
        highpass. 1D array_like, e.g. [10.0, 30.0] for bandpass or
        bandstop

    sfreq : float
        sampling frequency, e.g., 250.0, 500.0

    width_hz : float
        transition band width start to stop in Hz

    ripple_db : float
        attenuation in the stop band, in dB, e.g., 24.0, 60.0


    Returns
    -------
    taps : np.array
        coefficients of FIR filter.

    """

    for kwarg in [ftype, cutoff_hz, sfreq, width_hz, ripple_db, window]:
        assert kwarg is not None

    check_filter_params(
        ftype=ftype,
        cutoff_hz=cutoff_hz,
        sfreq=sfreq,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )

    # Nyquist frequency
    nyq_rate = sfreq / 2.0

    # transition band width in normalizied frequency
    width = width_hz / nyq_rate

    # order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    if N % 2 == 0:
        N = N + 1  # enforce odd number of taps

    # create a FIR filter using firwin
    if ftype.lower() == "lowpass":
        if window.lower() == "kaiser":
            taps = firwin(
                N, cutoff_hz, window=("kaiser", beta), pass_zero="lowpass", fs=sfreq,
            )
        else:
            taps = firwin(N, cutoff_hz, window=window, pass_zero="lowpass", fs=sfreq)
    elif ftype.lower() == "highpass":
        if window.lower() == "kaiser":
            taps = firwin(
                N, cutoff_hz, window=("kaiser", beta), pass_zero="highpass", fs=sfreq,
            )
        else:
            taps = firwin(N, cutoff_hz, window=window, pass_zero="highpass", fs=sfreq)
    elif ftype.lower() == "bandpass":
        if window.lower() == "kaiser":
            taps = firwin(
                N, cutoff_hz, window=("kaiser", beta), pass_zero="bandpass", fs=sfreq,
            )
        else:
            taps = firwin(N, cutoff_hz, window=window, pass_zero="bandpass", fs=sfreq)
    elif ftype.lower() == "bandstop":
        if window.lower() == "kaiser":
            taps = firwin(
                N, cutoff_hz, window=("kaiser", beta), pass_zero="bandstop", fs=sfreq,
            )
        else:
            taps = firwin(N, cutoff_hz, window=window, pass_zero="bandstop", fs=sfreq)

    return taps


def _sins_test_data(
    freq_list, amplitude_list, sampling_freq=250, duration=1.5, show_plot=False
):
    """creat a noisy signal to test the filter

    Parameters
    ----------
    freq_list : float, list

    amplitude_list : float, list

    sampling_freq : float, optional
        sampling frequency, default is 250.0

    duration : float, optional
        signal duration, default is 1.5 seconds

    Returns
    -------
    t,x : float
        time and values of a noisy signal  

    Examples
    --------
    >>> freq_list = [10.0, 25.0, 45.0]
    >>> amplitude_list = [1.0, 0.2, 0.3]
    >>> t, y = _sins_test_data(freq_list, amplitude_list)

    """
    assert len(freq_list) == len(amplitude_list)

    t = np.arange(0.0, duration, 1 / sampling_freq)
    x_noise = 0.1 * np.sin(2 * np.pi * 60 * t) + 0.2 * np.random.normal(size=len(t))
    # x = x_noise
    x = 0.0
    for i in range(len(freq_list)):
        x += amplitude_list[i] * np.sin(2 * np.pi * freq_list[i] * t)

    if show_plot:
        fig, ax = plt.subplots(figsize=(18, 4))
        ax.plot(t, x)

    return t, x


def _apply_firwin_filter_data(data, taps):
    """apply and phase compensate the FIRLS filtering to each column

    Parameters
    ----------
    data : array

    taps : ndarray
        Coefficients of FIR filter.

    Returns
    -------
    filtered_data : filtered data (same size as data)
        filtered array.

    """

    N = len(taps)
    delay = int((len(taps) - 1) / 2)
    a = 1.0

    msg = f"""
    applying linear phase delay compensated filter.
    a: {a}, N: {N}, delay: {delay}
    taps:
    {taps}
    """
    data = np.asanyarray(data).astype("float64")

    # add pads
    yy = []
    b = data[0:delay][::-1]
    e = data[-delay:][::-1]
    yy = np.append(b, data)
    yy = np.append(yy, e)

    # forward pass
    filtered_data = lfilter(taps, a, yy)

    # roll the phase shift by delay back to 0
    filtered_data = np.roll(filtered_data, -delay)[delay:-delay]

    if not len(data) == len(filtered_data):
        raise ValueError(
            f"filter I/O length mismatch: input={len(data)} output={len(filtered_data)}"
        )

    return filtered_data


# ------------------------------------------------------------
# public functions


def check_filter_params(
    ftype=None,
    cutoff_hz=None,
    sfreq=None,
    width_hz=None,
    ripple_db=None,
    window=None,
    allow_defaults=False,
):
    r"""type check FIR filter parameters and optionally provide defaults

    Values for `ftype`, `cutoff_hz`, and `sfreq` are obligatory.

    If `allow_defaults` is True, reasonable defaults are provided if
    any of window, width_hz and ripple_db are None.


    .. _filter_parameters_label:

    Parameters
    ----------
    ftype : str {'lowpass' , 'highpass', 'bandpass', 'bandstop'}
        filter type
    cutoff_hz : float or 1D-array-like of floats, length 2
        1/2 amplitude cutoff frequency in Hz
    sfreq : float
        sampling frequency, e.g., 250.0, 500.0
    width_hz : float
        pass-to-stop transition band width (Hz), symmetric for bandpass, bandstop
    ripple_db : float
        ripple, in dB, e.g., 53.0, 60.0
    window : str {'kaiser','hamming','hann','blackman'}
        window type for firwin
    allow_defaults : bool {False, True}
        If `True` this makes `width_hz`, `ripple_db`, `window` optional and
        fills in sensible defaults for any left unspecified by the user.


    Returns
    -------
    dict
       ``params`` with key:val for all filter parameters specified, suitable for passing as
       ``**params`` to spudtr.filters FIR functions.

    """

    def _test_numeric(_key, _val):
        """helper raises ValueError if _val is None or non-numeric"""
        try:
            if _val is None:
                raise Exception
            np.array(_val).astype(float)
        except:
            raise ValueError(f"{_key}={_val}, must be numeric")

    # ------------------------------------------------------------
    # API obligatory args: ftype, cutoff_hz, sfreq
    # ------------------------------------------------------------
    if ftype not in FTYPES:
        raise ValueError(f"ftype={ftype}, must be one of " + " ".join(FTYPES))

    for _param in ["cutoff_hz", "sfreq"]:
        _test_numeric(_param, eval(_param))

    # ------------------------------------------------------------
    # API optional args: window, width_hz, ripple_db
    # ------------------------------------------------------------
    if window is None and allow_defaults:
        window = "kaiser"
        warnings.warn(f"using default window='{window}'")

    if window not in WINDOWS:
        raise ValueError(f"window={window}, must be one of " + " ".join(WINDOWS))

    # compute default cutoff_hz and ripple_db for this window, ftype, sfreq
    _width_hz, _ripple_db = _trans_bwidth_ripple(
        ftype=ftype, cutoff_hz=cutoff_hz, sfreq=sfreq, window=window
    )

    if width_hz is None and allow_defaults:
        width_hz = _width_hz
        warnings.warn(f"using default width_hz={width_hz:0.3f}")

    if ripple_db is None and allow_defaults:
        ripple_db = _ripple_db
        warnings.warn(f"using default ripple_db={ripple_db:0.3f}")

    for _param in ["width_hz", "ripple_db"]:
        _test_numeric(_param, eval(_param))

    # load up return dict
    _params = {
        "ftype": ftype,
        "cutoff_hz": cutoff_hz,
        "width_hz": width_hz,
        "ripple_db": ripple_db,
        "window": window,
        "sfreq": sfreq,
    }
    assert all([val is not None for val in _params.values()])

    return _params


def show_filter(
    ftype=None,
    cutoff_hz=None,
    sfreq=None,
    width_hz=None,
    ripple_db=None,
    window=None,
    show_output=True,
):

    """Text summary and graphic display of filter attributes for the specified parameters.

    Figures are plotted for the transfer function, coefficients, and
    input-output performance on pure sine wave data with intervals of
    edge distortion highlighted.


    Parameters
    ----------
    ftype : str {'lowpass' , 'highpass', 'bandpass', 'bandstop'}
        filter type
    cutoff_hz : float or 1D array-like, length=2
        1/2 amplitude cutoff frequency (Hz)
    sfreq : float
        sampling frequency in samples per second
    width_hz : float, optional
        pass-to-stop transition band width (Hz)
    ripple_db : float, optional
       band ripple (dB)
    window : {'kaiser','hamming','hann','blackman'}, optional
        window type for firwin
    show_output : bool 
        plot example filter input-output, default=True


    Returns
    -------
    freq_phase : matplotlib.Figure
       plots frequency and phase response
    imp_resp: matplotlib.Figure
       plots impulse and step response
    s_edge : float
       number of seconds distorted at edge boundaries
    n_edge : int
       number of samples distorted at edge boundaries


    Notes
    -----
    For more information on the filter parameters see
    :ref:`check_filter params() Parameters <filter_parameters_label>`


    Examples
    --------
    >>> ftype = 'lowpass'
    >>> cutoff_hz = 10.0
    >>> sfreq = 250
    >>> width_hz = 5.0
    >>> ripple_db = 53.0
    >>> window = 'kaiser'

    >>> # fill in defaults for width_hz, ripple_db, window
    >>> show_filter(
          ftype=ftype,
          cutoff_hz=cutoff_hz,
          sfreq=sfreq
        )

    >>> # with all explicit parameters
    >>> show_filter(
          ftype=ftype,
          cutoff_hz=cutoff_hz,
          sfreq=sfreq,
          width_hz=width_hz,
          ripple_db=ripple_db,
          window=window,
        )

    """

    _fp = check_filter_params(
        ftype=ftype,
        cutoff_hz=cutoff_hz,
        sfreq=sfreq,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
        allow_defaults=True,
    )

    taps = _design_firwin_filter(**_fp)

    # this many samples are lost to edge distortion (worst case)
    n_edge = int(np.floor(len(taps) / 2.0))
    s_edge = n_edge / sfreq

    # promote scalar to iterable for printing
    _cutoff_hz = np.array(_fp["cutoff_hz"]).flatten()

    print(f"{_fp['ftype']} filter")
    print(f"sampling rate (samples / s): {_fp['sfreq']:0.3f}")
    print("1/2 amplitude cutoff (Hz): " + " ".join([f"{hz:0.3f}" for hz in _cutoff_hz]))
    print(f"transition width (Hz): {_fp['width_hz']:0.3f}")
    print(f"ripple (dB): {_fp['ripple_db']:0.3f}")
    print(f"window: {_fp['window']}")

    print(f"length (coefficients): {len(taps)}")
    print(f"delay (samples): {n_edge}")
    print(
        f"edge distortion: first and last {s_edge:.4f} seconds of the data"
        f"(= {n_edge} samples at {sfreq} samples / s)"
    )

    freq_phase = _mfreqz(
        b=taps,
        a=1,
        cutoff_hz=_fp["cutoff_hz"],
        sfreq=_fp["sfreq"],
        width_hz=_fp["width_hz"],
    )
    imp_step = _impz(b=taps, a=1)

    if show_output:
        io_fig, io_ax = filters_effect(**_fp)
        xdata_lims = np.array(
            [(l.get_xdata()[0], l.get_xdata()[-1]) for l in io_ax.get_lines()]
        ).max(axis=0)
        tmin, tmax = xdata_lims[0], xdata_lims[1]
        io_ax.axvspan(tmin, tmin + s_edge, color="gray", alpha=0.15)
        io_ax.axvspan(tmax, tmax - s_edge, color="gray", alpha=0.15)

    return freq_phase, imp_step, s_edge, n_edge


def fir_filter_dt(
    dt,
    col_names,
    ftype=None,
    cutoff_hz=None,
    sfreq=None,
    width_hz=None,
    ripple_db=None,
    window=None,
):

    """apply FIRLS filtering to columns of dataframe-like synchronized discrete time series

    Parameters
    ----------
    dt : pd.DataFrame or structured numpy nd.array with named data types 
        regularly sampled time-series data table: time (row) x data (columns)

    col_names: list of str
        column names to apply the transform

    key=val
        see :ref:`check_filter params() Parameters <filter_parameters_label>`


    Returns
    -------
    pd.DataFrame or np.ndarray
        table-like copy with filtered data columns, the same size and object type as dt


    Notes
    -----
    The input data is zero-padded by the length of the FIR filter delay and trimmed
    back to the original length.


    Examples
    --------
    >>> ftype = "bandpass"
    >>> cutoff_hz = [18, 35]
    >>> width_hz = 5
    >>> ripple_db = 60
    >>> window = "kaiser"
    >>> sfreq = 250

    >>> fir_filter_dt = epochs_filters(
        dt,
        col_names,
        ftype=ftype,
        window=window,
        cutoff_hz=cutoff_hz,
        width_hz=width_hz,
        ripple_db=ripple_db,
        sfreq=sfreq,
        trim_edges=False
    )

    >>> params = dict(ftype="lowpass", cutoff_hz=10, width_hz=5, ripple_db=60, sfreq=250, window="hamming")
    >>> fir_filter_dt = epochs_filters(dt, col_names, **params)
    """

    _fp = check_filter_params(
        ftype=ftype,
        cutoff_hz=cutoff_hz,
        sfreq=sfreq,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )

    taps = _design_firwin_filter(**_fp)

    # modicum of guarding
    if isinstance(dt, pd.DataFrame) or (
        isinstance(dt, np.ndarray) and dt.dtype.names is not None
    ):
        pass
    else:
        raise TypeError("dt must be pandas.DataFrame or structured numpy.ndarray")

    filt_dt = dt.copy()
    for column in col_names:

        filt_dt[column] = _apply_firwin_filter_data(dt[column], taps)

    return filt_dt


def fir_filter_data(
    data,
    ftype=None,
    cutoff_hz=None,
    sfreq=None,
    width_hz=None,
    ripple_db=None,
    window=None,
):

    """
    Finite Impulse Response filter

    Parameters
    ----------
    data : 1-D array-like

    key=val
        see :ref:`check_filter params() Parameters <filter_parameters_label>`

    Returns
    -------
    1D array
       ``filtered_data`` filter output, same length as ``data``

    Notes
    -----
    The input data is zero-padded by the length of the FIR filter delay and trimmed
    back to the original length.

    """

    _fp = check_filter_params(
        ftype=ftype,
        cutoff_hz=cutoff_hz,
        sfreq=sfreq,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )

    taps = _design_firwin_filter(**_fp)
    filt_data = _apply_firwin_filter_data(data, taps)
    return filt_data


def filters_effect(
    ftype=None, cutoff_hz=None, sfreq=None, width_hz=None, ripple_db=None, window=None,
):

    """
    Generate example filter input-output plots for pure sinewave data.


    Parameters
    ----------
    key=val
        see :ref:`check_filter params() Parameters <filter_parameters_label>`


    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
       ``fig``, ``ax`` of the example plot

    """

    # test signal lower and upper bounds
    LO_HZ_LB = 0.2
    HI_HZ_UB = sfreq / 2.0

    _fp = check_filter_params(
        ftype=ftype,
        cutoff_hz=cutoff_hz,
        sfreq=sfreq,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
        allow_defaults=False,
    )

    if isinstance(cutoff_hz, list):
        lo_hz = cutoff_hz[0] - width_hz
        hi_hz = cutoff_hz[1] + width_hz
    else:
        lo_hz = cutoff_hz - width_hz
        hi_hz = cutoff_hz + width_hz

    mid_hz = np.mean([lo_hz, hi_hz])  # same as np.mean(cutoff_hz)

    # bound lo, hi, mid hz
    lo_hz = np.max([LO_HZ_LB, lo_hz])
    hi_hz = np.min([HI_HZ_UB, hi_hz])
    assert lo_hz < mid_hz and mid_hz < hi_hz

    # set y, y1 sine wave lo_hz, hi_hz, w/ mid_hz for band pass/stop
    if ftype.lower() == "lowpass":
        y_freqs = [lo_hz, hi_hz]
        y1_freqs = [lo_hz]  # lo signal to pass

    elif ftype.lower() == "highpass":
        y_freqs = [lo_hz, hi_hz]
        y1_freqs = [hi_hz]  # hi signal to pass

    elif ftype.lower() == "bandpass":
        y_freqs = [lo_hz, mid_hz, hi_hz]
        y1_freqs = [mid_hz]  # in-band signal to pass

    elif ftype.lower() == "bandstop":
        y_freqs = [lo_hz, mid_hz, hi_hz]
        y1_freqs = [lo_hz, hi_hz]  # out-of-band signals to pass

    _fparams = dict(
        ftype=ftype,
        cutoff_hz=cutoff_hz,
        sfreq=sfreq,
        width_hz=width_hz,
        ripple_db=ripple_db,
        window=window,
    )

    # generate y, y1, and filter y
    y_amplitude_list = [1.0] * len(y_freqs)
    y1_amplitude_list = [1.0] * len(y1_freqs)

    # duration = 1/2 filter len + 3 cycles of low Hz + 1/2 filter len
    taps = _design_firwin_filter(**_fparams)
    duration = (3 * (1 / lo_hz)) + (len(taps) / sfreq)

    t, y = _sins_test_data(y_freqs, y_amplitude_list, sfreq, duration)
    t1, y1 = _sins_test_data(y1_freqs, y1_amplitude_list, sfreq, duration)
    y_filt = fir_filter_data(y, **_fparams)  # apply the filter

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t, y, ".-", color="c", linestyle="-", label="input")
    ax.plot(t, y1, ".-", color="b", linestyle="-", label="ideal output")
    ax.plot(
        t, y_filt, ".-", color="r", linestyle="-", label="%s filter output" % ftype,
    )

    # format for the title
    cutoff_hz_str = " ".join([f"{hz:.3f}" for hz in np.atleast_1d(cutoff_hz)])
    ax.set_title(
        (
            f"{ftype} filter cutoff={cutoff_hz_str} Hz, transition width={width_hz:.3f} Hz, "
            f"ripple={ripple_db:.3f}, dB window={window}"
        ),
        fontsize=20,
    )
    ax.set_xlabel("Time", fontsize=20)
    ax.legend(fontsize=16, loc="upper left", bbox_to_anchor=(1.05, 1.0))

    return fig, ax
