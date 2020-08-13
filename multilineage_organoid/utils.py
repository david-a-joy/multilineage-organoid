""" Utility functions used by multiple modules

* :py:func:`lowpass_filter`: Lowpass filter a signal with the filtfilt function
* :py:func:`calc_frequency_domain`: Convert a time domain signal to frequency

"""

# Imports
from typing import Tuple

# 3rd party
import numpy as np

from scipy.signal import butter, filtfilt, welch

# Our own imports
from .consts import FILTER_ORDER, FILTER_CUTOFF, DEBUG_OPTIMIZER

# Functions


def lowpass_filter(signals: np.ndarray,
                   sample_rate: float,
                   order: int = FILTER_ORDER,
                   cutoff: float = FILTER_CUTOFF):
    """ Lowpass filter the data

    :param ndarray signals:
        A t x k array of k signals with t timepoints
    :param float sample_rate:
        The sample rate for the signals
    :param int order:
        The order for the butterworth filter
    :param float cutoff:
        the cutoff in Hz for the filter
    """

    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    if normal_cutoff <= 0.0 or normal_cutoff >= 1.0:
        if DEBUG_OPTIMIZER:
            print(f'Cannot filter, got -3dB: {normal_cutoff}')
            print(f'Nyquist rate: {nyq}')
            print(f'Sample rate (Hz): {sample_rate}')
        return signals

    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    filtered_signals = []
    for i in range(signals.shape[1]):
        signal = signals[:, i]
        sigmask = ~np.isnan(signal)
        yf = filtfilt(b, a, signal[sigmask])

        yfinal = np.full_like(signal, np.nan)
        yfinal[sigmask] = yf

        filtered_signals.append(yfinal)
    return np.stack(filtered_signals, axis=1)


def calc_frequency_domain(time: np.ndarray,
                          signal: np.ndarray) -> Tuple[np.ndarray]:
    """ Calculate the frequency domain data

    :param ndarray time:
        The time array in seconds
    :param ndarray signal:
        The signal intensity
    :returns:
        The frequency array, the power at each frequency
    """
    dt = time[1] - time[0]
    sample_rate = 1.0 / dt
    xf, yf = welch(signal, fs=sample_rate)  # Welch's power estimate method
    return xf, 10.0 * np.log10(yf)
