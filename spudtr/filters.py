import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy import signal, fftpack

from pylab import *

import logging as LOGGER
from scipy.signal import kaiserord, firwin, freqz, lfilter


def suggest_epoch_length(sfreq, ripple_db, width_hz):

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


def show_filter(cutoff_hz, width_hz, ripple_db, sfreq, ftype, window):

    """
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
    window : string
        window type for firwin, e.g., 'kaiser','hamming','hann','blackman'

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

    taps = _design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )
    fig1 = _mfreqz(taps, sfreq, cutoff_hz, width_hz, a=1)
    fig2 = _impz(taps, a=1)


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
    h_dB = 20 * log10(abs(h))

    fig, (ax1, ax2) = plt.subplots(2, 1)
    # make a little extra space between the subplots
    fig.subplots_adjust(hspace=0.6)

    ax1.plot(w / max(w), h_dB, "b")
    ax1.set_ylim(-150, 5)
    ax1.set_ylabel("Magnitude (db)", color="b")
    ax1.set_xlabel(r"Normalized Frequency (x$\pi$rad/sample)")
    ax1.set_title(r"Frequency and Phase response")
    ax1b = ax1.twinx()
    h_Phase = unwrap(arctan2(imag(h), real(h)))
    ax1b.plot(w / max(w), h_Phase, "g")
    ax1b.set_ylabel("Phase (radians)", color="g")
    ax1.grid(linestyle="--")

    nyq_rate = sfreq / 2
    ax2.plot((w / np.pi) * nyq_rate, abs(h))
    cutoff_hz = np.atleast_1d(cutoff_hz)
    if cutoff_hz.size == 1:
        ax2.axvline(
            cutoff_hz + width_hz / 2, linestyle="--", linewidth=1, color="r"
        )
        ax2.axvline(
            cutoff_hz - width_hz / 2, linestyle="--", linewidth=1, color="r"
        )
    else:
        ax2.axvline(
            cutoff_hz[0] + width_hz / 2, linestyle="--", linewidth=1, color="r"
        )
        ax2.axvline(
            cutoff_hz[0] - width_hz / 2, linestyle="--", linewidth=1, color="r"
        )
        ax2.axvline(
            cutoff_hz[1] + width_hz / 2, linestyle="--", linewidth=1, color="r"
        )
        ax2.axvline(
            cutoff_hz[1] - width_hz / 2, linestyle="--", linewidth=1, color="r"
        )

    ax2.set_ylabel("Gain")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_title(r"Frequency Response")
    # ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlim(-0.05, 50)
    ax2.grid(linestyle="--")
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
    impulse = repeat(0.0, l)
    impulse[0] = 1.0
    x = arange(0, l)
    response = signal.lfilter(b, a, impulse)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # make a little extra space between the subplots
    fig.subplots_adjust(hspace=0.6)

    ax1.stem(x, response, use_line_collection=True)
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel(r"n (samples)")
    ax1.set_title(r"Impulse response")

    step = cumsum(response)
    ax2.stem(x, step, use_line_collection=True)
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel(r"n (samples)")
    ax2.set_title(r"Step response")
    return fig


def _design_firwin_filter(
    cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
):
    """
    FIRLS at https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

    Parameters
    ----------

    cutoff_hz : float or 1D array_like
        cutoff frequency in Hz, e.g., 5.0, 30.0 for lowpass or highpass. 1D array_like, e.g. [10.0, 30.0] for bandpass or bandstop

    width_hz : float
        transition band width start to stop in Hz

    ripple_db : float
        attenuation in the stop band, in dB, e.g., 24.0, 60.0

    sfreq : float
        sampling frequency, e.g., 250.0, 500.0

    ftype : string
        filter type, e.g., 'lowpass' , 'highpass', 'bandpass', 'bandstop'

    Returns
    -------
    taps : ndarray
        Coefficients of FIR filter.

    """

    # LOGGER.info(
    #    f"""
    # Buildiing firls filter: cutoff_hz={cutoff_hz}, width_hz={width_hz}, ripple_db={ripple_db}, sfreq={sfreq}
    # """
    # )

    # Nyquist frequency
    nyq_rate = sfreq / 2.0

    # transition band width in normalizied frequency
    width = width_hz / nyq_rate

    # order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    if N % 2 == 0:
        N = N + 1  # enforce odd number of taps

    # create a FIR filter using firwin .
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


def _apply_firwin_filter(df, columns, taps):
    """apply the FIRLS filtering

    Parameters
    ----------
    df : pd.DataFrame
        must have Epoch_idx and Time row index names

    columns: list of str
        column names to apply the filter

    taps : ndarray
        Coefficients of FIR filter.

    Returns
    -------
    filt_df : pd.DataFrame
        filtered df.
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

    filt_df = df.copy()
    for column in columns:

        # forward pass
        filtered_column = lfilter(taps, a, df[column])

        # roll the phase shift by delay back to 0
        filt_df[column] = np.roll(filtered_column, -delay)

    return filt_df


def epochs_filters(
    epochs_df,
    eeg_streams,
    ftype,
    window,
    cutoff_hz,
    width_hz,
    ripple_db,
    sfreq,
    trim_edges,
):
    """apply the FIRLS filtering for eeg data

    Parameters
    ----------
    epochs_df : pd.DataFrame
        must have Epoch_idx and Time row index names

    eeg_streams: list of str
        column names to apply the transform

    ftype : string
        filter type, e.g., 'lowpass' , 'highpass', 'bandpass', 'bandstop'

    window : string
        window type for firwin, e.g., 'kaiser','hamming','hann','blackman'

    cutoff_hz : float or 1D array_like
        cutoff frequency in Hz

    width_hz : float
        transition band width start to stop in Hz

    ripple_db : float
        attenuation in the stop band, in dB, e.g., 24.0, 60.0

    sfreq : float
        sampling frequency, e.g., 250.0, 500.0

    trim_edges : bool
        'True' trim edges, 'False' not trim edges

    Returns
    -------
    pd.DataFrame
        filtered epochs_df.

    Examples
    --------
    >>> ftype = "bandpass"
    >>> window = "kaiser"
    >>> cutoff_hz = [18, 35]
    >>> width_hz = 5
    >>> ripple_db = 60
    >>> sfreq = 250

    >>> filt_test_df = epochs_filters(
        epochs_df, 
        eeg_streams,
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

    >>> filt_test_df = epochs_filters(
        epochs_df, 
        eeg_streams,
        ftype,
        window,
        cutoff_hz,
        width_hz,
        ripple_db,
        sfreq,
        trim_edges=True
    )
    """

    # build and apply the filter
    taps = _design_firwin_filter(
        cutoff_hz, width_hz, ripple_db, sfreq, ftype, window
    )
    filt_epochs_df = _apply_firwin_filter(epochs_df, eeg_streams, taps)

    # optionally drop corrupted data
    if trim_edges:
        N = len(taps)
        half_width = int(np.floor(N / 2))
        # times = filt_epochs_df.index.unique("Time")
        times = filt_epochs_df.Time.unique()
        start_good = times[
            half_width
        ]  # == first good sample b.c. 0-base index
        stop_good = times[-(half_width + 1)]  # last good sample, 0-base index
        return filt_epochs_df.query(
            "Time >= @start_good and Time <= @stop_good"
        )
    else:
        return filt_epochs_df


def _sins_test_data(
    freq_list, amplitude_list, sampling_freq=None, duration=None
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
    >>> t,y = _sins_test_data(freq_list, amplitude_list)

    """
    assert len(freq_list) == len(amplitude_list)
    if sampling_freq is None:
        sampling_freq = 250
    if duration is None:
        duration = 1.5
    t = np.arange(0.0, duration, 1 / sampling_freq)
    x_noise = 0.1 * np.sin(2 * np.pi * 60 * t) + 0.2 * np.random.normal(
        size=len(t)
    )
    # x = x_noise
    x = 0.0
    for i in range(len(freq_list)):
        x += amplitude_list[i] * np.sin(2 * np.pi * freq_list[i] * t)
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t, x)
    return t, x
