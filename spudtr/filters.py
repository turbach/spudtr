import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy import signal, fftpack

import logging as LOGGER
from scipy.signal import kaiserord, firwin, freqz, lfilter


def _trans_bwidth_ripple(cutoff_hz, sfreq, ftype, window):

    """
    Parameters
    ----------
    cutoff_hz : float or 1D array_like
        cutoff frequency in Hz
    sfreq : float
        sampling frequency per second, e.g., 250.0, 500.0
    ftype : string
        filter type, e.g., 'lowpass' , 'highpass', 'bandpass', 'bandstop'
    window : string
        window type for firwin, e.g., 'kaiser','hamming','hann','blackman'

    Returns
    -------
    width_hz : float
        transition band width start to stop in Hz
    ripple_db : float
        attenuation in the stop band, in dB
    """

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


def _suggest_epoch_length(sfreq, ripple_db, width_hz):

    """
    Parameters
    ----------
    sfreq : float
        sampling frequency, i.e. 250.0
    ripple_db : float
        ripple in dB
    width_hz : float
        width of transition region in hz

    Examples
    --------
    >>> sfreq = 250
    >>> ripple_db = 60
    >>> width_hz = 4
    >>> suggest_epoch_length(sfreq, ripple_db, width_hz)
    your epoch length should be  230  points, or  0.92  seconds at least.
    """

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


def show_filter(
    cutoff_hz=None,
    sfreq=None,
    ftype=None,
    width_hz=None,
    ripple_db=None,
    window=None,
    show_output=True,
):

    """
    Parameters
    ----------
    cutoff_hz : float or 1D array_like
        cutoff frequency in Hz
    sfreq : float
        sampling frequency per second, e.g., 250.0, 500.0
    ftype : string
        filter type, e.g., 'lowpass' , 'highpass', 'bandpass', 'bandstop'
    width_hz : None or float
        transition band width start to stop in Hz
    ripple_db : None or float
        attenuation in the stop band, in dB, e.g., 24.0, 60.0
    window : None or string
        window type for firwin, e.g., 'kaiser','hamming','hann','blackman'
    show_output : True or False
        plot example filter input-output

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

    Examples
    --------
    >>> cutoff_hz = 10.0
    >>> width_hz = 5.0
    >>> ripple_db = 60.0
    >>> sfreq = 250
    >>> ftype = 'lowpass'
    >>> window = 'hamming'
    >>> show_filter(cutoff_hz, width_hz, ripple_db, sfreq, ftype, window)
    """

    if window is None:
        window = "kaiser"

    if width_hz is None or ripple_db is None:
        width_hz, ripple_db = _trans_bwidth_ripple(
            cutoff_hz, sfreq, ftype, window
        )

    taps = _design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )

    # this many samples are lost to edge distortion (worst case)
    n_edge = int(np.floor(len(taps) / 2.0))
    s_edge = n_edge / sfreq
    # print filter information
    print(f"{ftype} filter")
    print(f"sampling rate (samples / s): {sfreq:0.5f}")
    print(f"1/2 amplitude cutoff (Hz): {cutoff_hz:0.5f}")
    print(f"transition width (Hz): {width_hz:0.5f}")
    print(f"ripple (dB): {ripple_db:0.5f}")
    print(f"window: {window}")

    print(f"length (coefficients): {len(taps)}")
    print(f"delay (samples): {n_edge}")

    print(
        f"edge distortion: first and last {s_edge:.4f} seconds of the data"
        f"(= {n_edge} samples at {sfreq} samples / s)"
    )

    freq_phase = _mfreqz(taps, sfreq, cutoff_hz, width_hz, a=1)
    imp_step = _impz(taps, a=1)

    if show_output:
        io_fig, io_ax = filters_effect(
            cutoff_hz,
            sfreq,
            ftype,
            width_hz=width_hz,
            ripple_db=ripple_db,
            window=window,
        )
        tmin, tmax = io_ax.get_xlim()
        io_ax.axvspan(tmin, tmin + s_edge, color="gray", alpha=0.15)
        io_ax.axvspan(tmax, tmax - s_edge, color="gray", alpha=0.15)

    return freq_phase, imp_step, s_edge, n_edge


def _mfreqz(b, sfreq, cutoff_hz, width_hz, a=1):

    """ Plot the frequency and phase response of a digital filter.

    Parameters
    ----------
    cutoff_hz : float or 1D array_like
        cutoff frequency in Hz
    width_hz : float
        transition band width start to stop in Hz
    ripple_db : float
        attenuation in the stop band, in dB, e.g., 24.0, 60.0
    sfreq : float
        sampling frequency, e.g., 250.0, 500.0
    ftype : string
        filter type, e.g., 'lowpass' , 'highpass', 'bandpass', 'bandstop'
    b : array_like
        numerator of a linear filter
    a : array_like
        denominator of a linear filter

    Returns
    -------
    fig : `~.figure.Figure`
    """
    w, h = signal.freqz(b, a)
    h_dB = 20 * np.log10(abs(h))

    fig, (ax_freq, ax_freq_phase) = plt.subplots(2, 1)
    # make a little extra space between the subplots
    fig.subplots_adjust(hspace=0.6)

    # frequency response plot
    nyq_rate = sfreq / 2.0
    ax_freq.plot((w / np.pi) * nyq_rate, abs(h))
    cutoff_hz = np.atleast_1d(cutoff_hz)
    if cutoff_hz.size == 1:
        ax_freq.axvline(
            cutoff_hz + width_hz / 2, linestyle="--", linewidth=1, color="r"
        )
        ax_freq.axvline(
            cutoff_hz - width_hz / 2, linestyle="--", linewidth=1, color="r"
        )
    else:
        ax_freq.axvline(
            cutoff_hz[0] + width_hz / 2, linestyle="--", linewidth=1, color="r"
        )
        ax_freq.axvline(
            cutoff_hz[0] - width_hz / 2, linestyle="--", linewidth=1, color="r"
        )
        ax_freq.axvline(
            cutoff_hz[1] + width_hz / 2, linestyle="--", linewidth=1, color="r"
        )
        ax_freq.axvline(
            cutoff_hz[1] - width_hz / 2, linestyle="--", linewidth=1, color="r"
        )

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


def _impz(b, a=1):

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
    cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
):
    """calculate odd length, symmetric, linear phase FIR filter coefficients

    FIRLS at https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

    Parameters
    ----------

    cutoff_hz : float or 1D array_like
        cutoff frequency in Hz, e.g., 5.0, 30.0 for lowpass or
        highpass. 1D array_like, e.g. [10.0, 30.0] for bandpass or
        bandstop

    width_hz : float
        transition band width start to stop in Hz

    ripple_db : float
        attenuation in the stop band, in dB, e.g., 24.0, 60.0

    sfreq : float
        sampling frequency, e.g., 250.0, 500.0

    ftype : string
        filter type, one of 'lowpass' , 'highpass', 'bandpass', 'bandstop'

    Returns
    -------
    taps : np.array
        coefficients of FIR filter.

    """

    # a bit of guarding
    for _arg in [
        "cutoff_hz",
        "width_hz",
        "ripple_db",
        "sfreq",
        "ftype",
        "window",
    ]:
        if eval(_arg) is None:
            raise ValueError(f"{_arg} is None, set a value")

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
                N,
                cutoff_hz,
                window=("kaiser", beta),
                pass_zero="lowpass",
                fs=sfreq,
            )
        else:
            taps = firwin(
                N, cutoff_hz, window=window, pass_zero="lowpass", fs=sfreq
            )
    elif ftype.lower() == "highpass":
        if window.lower() == "kaiser":
            taps = firwin(
                N,
                cutoff_hz,
                window=("kaiser", beta),
                pass_zero="highpass",
                fs=sfreq,
            )
        else:
            taps = firwin(
                N, cutoff_hz, window=window, pass_zero="highpass", fs=sfreq
            )
    elif ftype.lower() == "bandpass":
        if window.lower() == "kaiser":
            taps = firwin(
                N,
                cutoff_hz,
                window=("kaiser", beta),
                pass_zero="bandpass",
                fs=sfreq,
            )
        else:
            taps = firwin(
                N, cutoff_hz, window=window, pass_zero="bandpass", fs=sfreq
            )
    elif ftype.lower() == "bandstop":
        if window.lower() == "kaiser":
            taps = firwin(
                N,
                cutoff_hz,
                window=("kaiser", beta),
                pass_zero="bandstop",
                fs=sfreq,
            )
        else:
            taps = firwin(
                N, cutoff_hz, window=window, pass_zero="bandstop", fs=sfreq
            )

    return taps


def _sins_test_data(
    freq_list,
    amplitude_list,
    sampling_freq=250,
    duration=1.5,
    show_plot=False,
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
    x_noise = 0.1 * np.sin(2 * np.pi * 60 * t) + 0.2 * np.random.normal(
        size=len(t)
    )
    # x = x_noise
    x = 0.0
    for i in range(len(freq_list)):
        x += amplitude_list[i] * np.sin(2 * np.pi * freq_list[i] * t)

    if show_plot:
        fig, ax = plt.subplots(figsize=(18, 4))
        ax.plot(t, x)

    return t, x


def fir_filter_dt(
    dt,
    col_names,
    cutoff_hz=None,
    sfreq=None,
    ftype=None,
    width_hz=None,
    ripple_db=None,
    window=None,
):

    """apply FIRLS filtering to columns of synchronized discrete time series

    Note
    ----
    The `trim_edges` option crops the head and tail of the entire data frame, not 
    just the selected columns.

    Parameters
    ----------
    dt : pd.DataFrame or np.ndarray with named dtypes
        regularly sampled time-series data table: time (row) x data (columns)

    col_names: list of str
        column names to apply the transform

    cutoff_hz : float or 1D array_like
        cutoff frequency in Hz

    sfreq : float
        sampling frequency, e.g., 250.0, 500.0

    ftype : string
        filter type, e.g., 'lowpass' , 'highpass', 'bandpass', 'bandstop'

    width_hz : None or float
        transition band width start to stop in Hz

    ripple_db : None or float
        attenuation in the stop band, in dB, e.g., 24.0, 60.0

    window : None or string
        window type for firwin, e.g., 'kaiser','hamming','hann','blackman'


    Returns
    -------
    dt : 
        data table with filtered data columns, the same type object as input

    Examples
    --------
    >>> ftype = "bandpass"
    >>> window = "kaiser"
    >>> cutoff_hz = [18, 35]
    >>> width_hz = 5
    >>> ripple_db = 60
    >>> sfreq = 250

    >>> filt_test_dt = epochs_filters(
        dt,
        col_names,
        ftype,
        window,
        cutoff_hz,
        width_hz,
        ripple_db,
        sfreq,
        trim_edges=False
    )

    >>> ftype = "lowpass"
    >>> window = "hamming"
    >>> cutoff_hz = 10
    >>> width_hz = 5
    >>> ripple_db = 60
    >>> sfreq = 250

    >>> filt_test_dt = epochs_filters(
            dt,
            col_names,
            cutoff_hz,
            sfreq,
            ftype,
            width_hz=None,
            ripple_db=None,
            window=None,
        )
    """
    if window is None:
        window = "kaiser"

    if width_hz is None or ripple_db is None:
        width_hz, ripple_db = _trans_bwidth_ripple(
            cutoff_hz, sfreq, ftype, window
        )

    # modicum of guarding
    if isinstance(dt, pd.DataFrame):
        pass
    else:
        raise TypeError("dt must be pandas.DataFrame or np.ndarray")

    # build and apply the filter
    taps = _design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )

    filt_dt = dt.copy()
    for column in col_names:

        filt_dt[column] = _apply_firwin_filter_data(dt[column], taps)

    return filt_dt


def fir_filter_data(
    data,
    cutoff_hz=None,
    sfreq=None,
    ftype=None,
    width_hz=None,
    ripple_db=None,
    window=None,
):

    """
    Finite Impulse Response filter

    Parameters
    ----------
    data : 1-D array
    cutoff_hz : float or 1D array_like
        cutoff frequency in Hz
    sfreq : float
        sampling frequency per second, e.g., 250.0, 500.0
    ftype : string
        filter type, e.g., 'lowpass' , 'highpass', 'bandpass', 'bandstop'
    width_hz : float
        transition band width start to stop in Hz
    ripple_db : float
        attenuation in the stop band, in dB, e.g., 24.0, 60.0
    window : string
        window type for firwin, e.g., 'kaiser','hamming','hann','blackman'

    Returns
    -------
        filt_data : filtered data
    """

    if window is None:
        window = "kaiser"

    if width_hz is None or ripple_db is None:
        width_hz, ripple_db = _trans_bwidth_ripple(
            cutoff_hz, sfreq, ftype, window
        )

    # build and apply the filter
    taps = _design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )

    filt_data = _apply_firwin_filter_data(data, taps)

    return filt_data


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

    # assert len(taps) % 2 == 1  # enforce odd number of taps

    N = len(taps)

    delay = int((len(taps) - 1) / 2)
    a = 1.0

    msg = f"""
    applying linear phase delay compensated filter.
    a: {a}, N: {N}, delay: {delay}
    taps:
    {taps}
    """
    # LOGGER.info(msg)
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
        raise ValueError("The input data is too short.")

    return filtered_data


def filters_effect(
    cutoff_hz=None,
    sfreq=None,
    ftype=None,
    width_hz=None,
    ripple_db=None,
    window=None,
):

    """
    Parameters
    ----------
    cutoff_hz : float or 1D array_like
        cutoff frequency in Hz
    sfreq : float
        sampling frequency per second, e.g., 250.0, 500.0
    ftype : string
        filter type, e.g., 'lowpass' , 'highpass', 'bandpass', 'bandstop'
    width_hz : float
        transition band width start to stop in Hz
    ripple_db : float
        attenuation in the stop band, in dB
    window : string
        window type for firwin, e.g., 'kaiser','hamming','hann','blackman'

    Returns
    -------
    fig, ax : `~.figure.Figure`, `~axes.Axes`
    -------
    """

    # test signal lower and upper bounds
    LO_HZ_LB = 0.2
    HI_HZ_UB = sfreq / 2.0

    duration = 1.5  # seconds, default may be overriden

    if window is None:
        window = "kaiser"

    if width_hz is None or ripple_db is None:
        width_hz, ripple_db = _trans_bwidth_ripple(
            cutoff_hz, sfreq, ftype, window
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
        if lo_hz == LO_HZ_LB:
            duration = 5
        y_freqs = [lo_hz, hi_hz]
        y1_freqs = [lo_hz]  # lo signal to pass

    elif ftype.lower() == "highpass":
        y_freqs = [lo_hz, hi_hz]
        y1_freqs = [hi_hz]  # hi signal to pass

    elif ftype.lower() == "bandpass":
        y_freqs = [lo_hz, mid_hz, hi_hz]
        y1_freqs = [mid_hz]  # in-band signal to pass

    elif ftype.lower() == "bandstop":
        if lo_hz == LO_HZ_LB:
            duration = 5
        y_freqs = [lo_hz, mid_hz, hi_hz]
        y1_freqs = [lo_hz, hi_hz]  # out-of-band signals to pass

    # generate y, y1, and filter y
    y_amplitude_list = [1.0] * len(y_freqs)
    y1_amplitude_list = [1.0] * len(y1_freqs)

    t, y = _sins_test_data(y_freqs, y_amplitude_list, sfreq, duration)
    t1, y1 = _sins_test_data(y1_freqs, y1_amplitude_list, sfreq, duration)
    y_filt = fir_filter_data(
        y, cutoff_hz, sfreq, ftype, width_hz, ripple_db, window
    )

    fig, ax = plt.subplots(figsize=(16, 4))

    ax.plot(t, y, ".-", color="c", linestyle="-", label="input")
    ax.plot(t, y1, ".-", color="b", linestyle="-", label="ideal output")
    ax.plot(
        t,
        y_filt,
        ".-",
        color="r",
        linestyle="-",
        label="%s filter output" % ftype,
    )
    ax.set_title(
        (
            f"{ftype} filter cutoff={cutoff_hz} Hz, transition width={width_hz} Hz, "
            f"ripple={ripple_db}, dB window={window}"
        ),
        fontsize=20,
    )
    ax.set_xlabel("Time", fontsize=20)
    ax.legend(fontsize=16, loc="upper left", bbox_to_anchor=(1.05, 1.0))
    return fig, ax
