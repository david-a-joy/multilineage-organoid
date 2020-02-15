""" Signal processing for calcium trace data

Main processing functions:

* :py:func:`filter_datafile`: Filter all signals in a single data file
* :py:func:`maybe_analyze_datafile`: Multiprocessing wrapper to :py:func:`filter_datafile`

Classes:

* :py:class:`AnalysisParams`: Analysis parameters for each trace study
* :py:class:`ExpModelFit`: Fit decay curves to traces

Functions:

* :py:func:`fit_model`: Fit decay curves to traces

Signal Helpers:

* :py:func:`calc_frequency_domain`: Convert from time to frequency domain

"""

# Imports
import pathlib
import traceback
from dataclasses import dataclass
from typing import List, Optional

# 3rd party libs
import numpy as np

import matplotlib.pyplot as plt

from skimage.feature import peak_local_max

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor, LinearRegression, Ridge

from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, welch

# Our own imports
from .io import DataReader, save_outfile
from .consts import (
    SIGNAL_TYPE, FILTER_CUTOFF, SAMPLES_AROUND_PEAK, LINEAR_MODEL, DATA_TYPE,
    TIME_SCALE, PLOT_SUFFIX, FILTER_ORDER, DEBUG_OPTIMIZER, MIN_STATS_SCORE,
    FIGSIZE
)

# Classes


@dataclass
class AnalysisParams(object):
    """ Parameter object to pass to the multiprocessing system

    :param Path datafile:
        The raw trace '.csv' or '.xlsx' file to analyze
    :param list[str] plot_types:
        The list of debugging plots to render ('all' for all, 'none' to show no plots)
    :param str signal_type:
        Which normalization method to use (one of 'F/F0', 'F-F0', 'F-F0/F0')
    :param float filter_cutoff:
        -3dB point for the filter (Hz)
    :param int filter_order:
        The order of the lowpass butterworth filter
    :param bool skip_detrend:
        Skip the linear detrending step
    :param bool skip_lowpass:
        Skip the lowpass filter step
    :param bool skip_model_fit:
        Skip the model fit for the decay curve step
    :param int samples_around_peak:
        The number of samples around each maxima in the signal
    :param bool flip_signal:
        If True, flip the signal (max becomes min, and vice versa)
    :param str level_shift:
        How to shift the raw data when doing F/F0 normalization (default 'min')
    :param float time_scale:
        The conversion factor from samples to seconds
    :param str linear_model:
        Which model to use to fit for detrending
    :param Path plotdir:
        The path to save plots to
    :param str plot_suffix:
        The suffix to save plots with (either '.png' or '.svg')
    """

    datafile: pathlib.Path
    plot_types: List[str]

    signal_type: str = SIGNAL_TYPE
    filter_cutoff: float = FILTER_CUTOFF
    filter_order: int = FILTER_ORDER

    skip_detrend: bool = False
    skip_lowpass: bool = False
    skip_model_fit: bool = False

    samples_around_peak: int = SAMPLES_AROUND_PEAK
    data_type: str = DATA_TYPE

    flip_signal: bool = False
    level_shift: Optional[str] = None
    time_scale: float = TIME_SCALE

    linear_model: str = LINEAR_MODEL

    plotdir: Optional[pathlib.Path] = None
    plot_suffix: str = PLOT_SUFFIX


class ExpModelFit(object):
    """ Fit different exponential models

    :param ndarray time:
        Time vector for the signal
    :param ndarray signal:
        Values for the signal
    :param float time_scale:
        Samples/second of the signal
    """

    def __init__(self,
                 time: np.ndarray,
                 signal: np.ndarray,
                 time_scale: float = TIME_SCALE):
        self.time = time
        self.signal = signal
        self.time_scale = time_scale

        # Tau as ms (should be approximately t90_down - t_peak)
        self._scaled_time = None
        self._smooth_signal = None

        self.amp_min = 0.0
        self.tc_min = 0.0
        self.offset_min = -4.0

        self.amp_max = None
        self.tc_max = 10.0
        self.offset_max = 10.0
        self.max_feval = 5000

        # Initial guesses
        self.init_tc = None
        self.init_amp = None
        self.init_offset = None

        # Single fit
        self.se_tc = None
        self.se_amp = None
        self.se_offset = None

        # Double fit
        self.de_amp = None
        self.de_offset = None
        self.de_tc1 = None
        self.de_tc2 = None
        self.de_tmean = None
        self.de_tsigma = None

    def smooth_signal(self):
        """ Smooth the signal using legendre polynomials """

        scaled_time = np.linspace(0, 1, self.signal.shape[0])
        mask = np.logical_and(np.isfinite(self.signal), np.isfinite(scaled_time))
        self._smooth_signal = self.signal[mask]
        self._scaled_time = scaled_time[mask]
        self.se_offset = self.de_offset = np.min(self._smooth_signal)

    def initialize_guesses(self):
        """ Initial guess for the signal """
        self.init_offset = np.min(self._smooth_signal)
        log_signal = np.log(self._smooth_signal - np.min(self._smooth_signal) + 1)
        coeffs = np.polyfit(self._scaled_time, log_signal, 1)

        self.init_tc = -coeffs[0]
        self.init_amp = coeffs[1]
        self.amp_max = 1.5*(np.max(self._smooth_signal) - np.min(self._smooth_signal))

        if DEBUG_OPTIMIZER:
            print('Initial Tc Guess:     {:0.4f}'.format(self.init_tc))
            print('Initial Amp Guess:    {:0.4f}'.format(self.init_amp))
            print('Initial Offset Guess: {:0.4f}'.format(self.init_offset))

    def fit_single_exp(self):
        """ Fit a single exponential curve """
        # Guess at the initial parameters and boundaries
        param_guesses = (self.init_amp, self.init_tc, self.init_offset)
        param_bounds = [(self.amp_min, self.tc_min, self.offset_min),
                        (self.amp_max, self.tc_max, self.offset_max)]
        try:
            fit_opt, _ = curve_fit(
                self.single_exp_model, self._scaled_time, self._smooth_signal,
                p0=param_guesses,
                bounds=param_bounds,
                maxfev=self.max_feval,
            )
        except (ValueError, RuntimeError):
            if DEBUG_OPTIMIZER:
                print('Single opt failed...')
                traceback.print_exc()
            # If we get a fit error, just return nans
            fit_opt = tuple(np.nan for _ in param_guesses)
        self.se_amp = fit_opt[0]
        self.se_tc = fit_opt[1]
        self.se_offset = fit_opt[2]

        if DEBUG_OPTIMIZER:
            print('Single Tc Fit:     {:0.4f}'.format(self.se_tc))
            print('Single Amp Fit:    {:0.4f}'.format(self.se_amp))
            print('Single Offset Fit: {:0.4f}'.format(self.se_offset))

    def fit_double_exp(self):
        """ Fit a double exponential curve """
        # Guess at the initial parameters and boundaries
        # amp, offset, tc1, tc2, tmean, tsigma
        param_guesses = (self.se_amp, self.se_offset, self.se_tc*0.5, self.se_tc*1.5, 0.5, 0.1)
        param_bounds = [(self.amp_min, self.offset_min, self.tc_min, self.tc_min, 0.1, 0.01),
                        (self.amp_max, self.offset_max, self.tc_max, self.tc_max, 0.9, 0.2)]
        try:
            fit_opt, _ = curve_fit(
                self.double_exp_model, self._scaled_time, self._smooth_signal,
                p0=param_guesses,
                bounds=param_bounds,
                maxfev=self.max_feval,
            )
        except (ValueError, RuntimeError):
            if DEBUG_OPTIMIZER:
                print('Double opt failed...')
                traceback.print_exc()
            # If we get a fit error, just return nans
            fit_opt = tuple(np.nan for _ in param_guesses)

        # amp, offset, tc1, tc2, tmean, tsigma
        self.de_amp = fit_opt[0]
        self.de_offset = fit_opt[1]
        self.de_tc1 = fit_opt[2]
        self.de_tc2 = fit_opt[3]
        self.de_tmean = fit_opt[4]
        self.de_tsigma = fit_opt[5]

        if DEBUG_OPTIMIZER:
            print('Double Amp Fit:    {:0.4f}'.format(self.de_amp))
            print('Double Offset Fit: {:0.4f}'.format(self.de_offset))
            print('Double Tc1 Fit:    {:0.4f}'.format(self.de_tc1))
            print('Double Tc2 Fit:    {:0.4f}'.format(self.de_tc2))
            print('Double Tmean Fit:  {:0.4f}'.format(self.de_tmean))
            print('Double Tsigma Fit: {:0.4f}'.format(self.de_tsigma))

    def plot_fit(self):
        """ Plot the model fit """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))

        se_signal = self.single_exp_model(self._scaled_time, self.se_amp, self.se_tc, self.se_offset)
        de_signal = self.double_exp_model(
            self._scaled_time, self.de_amp, self.de_offset, self.de_tc1, self.de_tc2,
            self.de_tmean, self.de_tsigma)

        ax1.plot(self._scaled_time, se_signal, '-r')
        ax1.plot(self._scaled_time, self._smooth_signal, 'om')

        ax2.plot(self._scaled_time, de_signal, '-r')
        ax2.plot(self._scaled_time, self._smooth_signal, 'om')

        fig.suptitle('{} Points'.format(se_signal.shape[0]))

        plt.show()

    def get_single_exp_params(self):
        """ Get a dictionary of parameters for the single exp fit

        :returns:
            A dictionary of the parameters, converted back to real time
        """
        time_range = np.nanmax(self.time) - np.nanmin(self.time)
        return {
            'se_amp': self.se_amp,
            'se_offset': self.se_offset,
            'se_tc': time_range / self.se_tc,
        }

    def get_double_exp_params(self):
        """ Get a dictionary of parameters for the single exp fit

        :returns:
            A dictionary of the parameters, converted back to real time
        """
        time_range = np.nanmax(self.time) - np.nanmin(self.time)
        return {
            'de_amp': self.de_amp,
            'de_offset': self.de_offset,
            'de_tc1': time_range / self.de_tc1,
            'de_tc2': time_range / self.de_tc2,
            'de_tmean': self.de_tmean,
            'de_tsigma': self.de_tsigma,
        }

    @staticmethod
    def double_exp_model(x: np.ndarray,
                         amp: float,
                         offset: float,
                         tc1: float,
                         tc2: float,
                         tmean: float,
                         tsigma: float):
        """ Double exponential model

        Based on:
        "Piecewise exponential survival curves with smooth transitions"
        Zelterman et al, Mathematical Biosciences. 1994.

        :param ndarray x:
            The time signal
        :param float amp:
            The amplitude for the system
        :param float offset:
            The signal offset for the system
        :param float tc1:
            The time constant for x < tmean
        :param float tc2:
            The time constant for x > tmean
        :param float tmean:
            The mean transition point between tc1 and tc2
        :param float tsigma:
            The width of the transition between tc1 and tc2 around tmean
        :returns:
            The exponential decay curve
        """
        # Specifically, this implements eq 9
        mask = x <= tmean - tsigma
        signal1 = np.exp(-tc1*x)
        signal2 = np.exp(-tc2*x - (tc1-tc2)*(tmean - tsigma*np.exp(-(x - tmean + tsigma)/tsigma)))
        signal = np.zeros_like(x)
        signal[mask] = signal1[mask] * amp + offset
        signal[~mask] = signal2[~mask] * amp + offset
        return signal

    @staticmethod
    def single_exp_model(x: np.ndarray,
                         amp: float,
                         tc: float,
                         offset: float):
        """ Single exponential signal fit model

        :param ndarray x:
            The time signal
        :param float amp:
            The amplitude
        :param float tc:
            The time constant
        :param float offset:
            The signal offset
        :returns:
            The exponential decay curve
        """
        return amp*np.exp(-tc*x) + offset


# Functions


def calc_frequency_domain(time: np.ndarray,
                          signal: np.ndarray):
    """ Calculate the frequency domain data

    :param ndarray time:
        The time array in seconds
    :param ndarray signal:
        The signal intensity
    :returns:
        The frequency array, the power at each frequency
    """
    dt = time[1] - time[0]
    fs = 1 / dt
    xf, yf = welch(signal, fs=fs)  # Welch's power estimate method
    return xf, 10 * np.log10(yf)


def fit_model(time: np.ndarray,
              signal: np.ndarray,
              time_scale: float = TIME_SCALE) -> ExpModelFit:
    """ Fit a model to the signal data

    :param ndarray time:
        The 1D time array
    :param ndarray signal:
        The 1D signal array
    :param func model:
        The model function to fit
    :param float time_scale:
        The scale factor for time (milliseconds/second)
    :returns:
        A tuple of model parameters for each model
    """
    model = ExpModelFit(time, signal,
                        time_scale=time_scale)
    model.smooth_signal()
    model.initialize_guesses()
    model.fit_single_exp()
    model.fit_double_exp()
    return model


def plot_signals(infile: pathlib.Path,
                 time: np.ndarray,
                 signals: np.ndarray,
                 plot_types: Optional[List[str]] = None,
                 raw_signals: Optional[np.ndarray] = None,
                 trend_lines=None,
                 stats=None,
                 signal_type=SIGNAL_TYPE,
                 time_scale=TIME_SCALE,
                 filter_cutoff=FILTER_CUTOFF,
                 figsize=FIGSIZE,
                 plotfile=None):
    """ Plot the signals over time

    :param Path infile:
        The name of the input CSV file
    :param ndarray time:
        The n x 1 time vector
    :param ndarray signals:
        The n x m array of signal vectors
    :param list[str] plot_types:
        The list of different plots to make
    :param ndarray raw_signals:
        If not None, the raw signals to plot
    :param dict stats:
        The dictionary of peak stats for this signal
    :param str signal_type:
        The normalization method used on the signal
    :param float time_scale:
        The scaling from samples to seconds
    :param float filter_cutoff:
        The -3dB point on the filter (in Hz)
    :param Path plotfile:
        The path to save the plot to or None to show the plot
    """

    if plot_types in (None, []):
        return
    if isinstance(plot_types, str):
        plot_types = [plot_types]
    if any([p in ('none', None) for p in plot_types]):
        return

    if stats is None:
        stats = [{} for _ in range(signals.shape[1])]
    if len(stats) != signals.shape[1]:
        err = 'Got {} stat measurements but {} signals'
        err = err.format(len(stats), signals.shape[1])
        raise ValueError(err)

    if raw_signals is not None:
        # Do a quick normalization to keep everything on the same scale
        raw_mean = np.nanmean(raw_signals, axis=0, keepdims=True)
        assert raw_mean.shape[1] == raw_signals.shape[1]
        if signal_type == 'F/F0':
            raw_signals = raw_signals / raw_mean
        elif signal_type == 'F-F0/F0':
            raw_signals = (raw_signals - raw_mean) / raw_mean
        elif signal_type == 'F-F0':
            raw_signals = raw_signals - raw_mean
        else:
            raise KeyError('Unknown signal type: {}'.format(signal_type))
        assert raw_signals.shape[0] == time.shape[0]
        assert raw_signals.shape[1] == signals.shape[1]

        # Also normalize the trend line
        if trend_lines is not None:
            if signal_type == 'F/F0':
                trend_lines = trend_lines / raw_mean
            elif signal_type == 'F-F0/F0':
                trend_lines = (trend_lines - raw_mean) / raw_mean
            elif signal_type == 'F-F0':
                trend_lines = trend_lines - raw_mean
            else:
                raise KeyError('Unknown signal type: {}'.format(signal_type))
            assert trend_lines.shape[1] == signals.shape[1]

    for i in range(signals.shape[1]):
        if 'freq' in plot_types and ('raw' in plot_types or 'filtered' in plot_types):
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(figsize*2, figsize))
        elif 'freq' in plot_types:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize, figsize))
        else:
            fig, ax0 = plt.subplots(1, 1, figsize=(figsize, figsize))

        # Time domain signal
        if 'filtered' in plot_types:
            ax0.plot(time, signals[:, i], 'r', label='filtered', linewidth=2)
        if raw_signals is not None and 'raw' in plot_types:
            ax0.plot(time, raw_signals[:, i], '--b', label='raw')
            ax0.plot(time, trend_lines[:, i], '-c', linewidth=2, label='trend')

        # Plot the processed data
        if 'filtered' in plot_types:
            signal_stats = stats[i].get('signal_stats', [])
            for peak_stats in signal_stats:
                # Plot the detected peaks and the surrounding mins
                peak_index = peak_stats['peak_index']
                peak_start_index = peak_stats['peak_start_index']
                peak_end_index = peak_stats['peak_end_index']

                ax0.plot(time[peak_index], signals[peak_index, i], 'ro')
                ax0.plot([time[peak_start_index], time[peak_end_index]],
                         [signals[peak_start_index, i], signals[peak_end_index, i]],
                         'go--')

                # Plot the exponential fits
                peak_time = time[peak_index:peak_end_index]
                peak_time_offset = np.min(peak_time)
                peak_time_end = np.max(peak_time)
                peak_range = (peak_time_end - peak_time_offset)
                scaled_peak_time = (peak_time - peak_time_offset)/peak_range

                # Single exponential fit
                if 'se_amp' in peak_stats:
                    scaled_peak_fit = ExpModelFit.single_exp_model(scaled_peak_time,
                                                                   peak_stats['se_amp'],
                                                                   peak_range/peak_stats['se_tc'],
                                                                   peak_stats['se_offset'])
                    if np.any(np.isfinite(scaled_peak_fit)):
                        ax0.plot(peak_time + peak_time_offset/time_scale,
                                 scaled_peak_fit, '-g')
                    elif DEBUG_OPTIMIZER:
                        print('Invalid single fit for {}'.format(infile.stem))

                # Double exponential fit
                if 'de_amp' in peak_stats:
                    scaled_peak_fit = ExpModelFit.double_exp_model(
                        scaled_peak_time, peak_stats['de_amp'], peak_stats['de_offset'],
                        peak_range/peak_stats['de_tc1'], peak_range/peak_stats['de_tc2'],
                        peak_stats['de_tmean'], peak_stats['de_tsigma'],
                    )
                    if np.any(np.isfinite(scaled_peak_fit)):
                        ax0.plot(peak_time + peak_time_offset/time_scale,
                                 scaled_peak_fit, '-', color='#00AA00')
                    elif DEBUG_OPTIMIZER:
                        print('Invalid double fit for {}'.format(infile.stem))

        if 'raw' in plot_types or 'filtered' in plot_types:
            ax0.set_xlabel('Time (ms)')
            ax0.set_ylabel('Intensity')
            ax0.set_title('{} Intensity {}'.format(signal_type, infile.stem))
            ax0.legend()

        # Frequency domain signal
        if 'freq' in plot_types:
            filt_x, filt_y = calc_frequency_domain(time / time_scale, signals[:, i])
            ax1.plot(filt_x, filt_y, 'r', label='filtered', linewidth=2)
            if raw_signals is not None:
                raw_x, raw_y = calc_frequency_domain(time / time_scale, raw_signals[:, i])
                ax1.plot(raw_x, raw_y, '--b', label='raw')

            ax1.axvline(x=filter_cutoff, ymin=0, ymax=1, color='g',
                        label='Filter -3dB')

            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Intensity (dB)')
            ax1.set_title('{} Intensity {}'.format(signal_type, infile.stem))
            ax1.legend()

        if plotfile is None:
            plt.show()
        else:
            plotfile = pathlib.Path(plotfile)
            final_plotfile = plotfile.parent / '{}_{:02d}{}'.format(plotfile.stem, i+1, plotfile.suffix)
            final_plotfile.parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(str(final_plotfile))
            plt.close()


def remove_trend(signals, signal_type=SIGNAL_TYPE, skip_detrend=False, linear_model=LINEAR_MODEL):
    """ Detrend the signals because they have bad trend lines

    Uses a RANSAC linear regression to fit a line to the base of the signal

    :param ndarray signals:
        The array of signals to detrend
    :param str signal_type:
        Which normalization method to use (one of 'F/F0', 'F-F0', 'F-F0/F0')
    :param bool skip_detrend:
        If True, skip the linear detrend pass and just normalize to the signal minimum
    :returns:
        A new ndarray with the signals normalized
    """

    linear_model = linear_model.lower()
    trend_lines = []
    detrended_signals = []

    for i in range(signals.shape[1]):
        signal = signals[:, i]
        sigmask = ~np.isnan(signal)

        # Drop any nan values because the signals aren't all aligned
        x = np.arange(signal.shape[0])
        xm = x[sigmask]
        ym = signal[sigmask]

        if skip_detrend:
            # No linear detrending, normalize to the minimum
            yr = np.full_like(x, np.min(ym))
        elif linear_model == 'boxcar':
            yr = filtfilt([1] * 200, [200], ym)
        elif linear_model == 'least_squares':
            model = LinearRegression()
            model.fit(xm.reshape(-1, 1), ym)
            yr = model.predict(x.reshape(-1, 1))
        elif linear_model == 'ransac':
            # Fit a robust line to the signal
            # Works better than a classic least squares fit
            model = RANSACRegressor(LinearRegression())
            model.fit(xm.reshape(-1, 1), ym)
            yr = model.predict(x.reshape(-1, 1))
        elif linear_model == 'exp_ransac':
            # Do an exponential fit for the data
            model = RANSACRegressor(LinearRegression())
            model.fit(xm.reshape(-1, 1), np.log(ym))
            yr = np.exp(model.predict(x.reshape(-1, 1)))
        elif linear_model == 'quadratic':
            model = make_pipeline(PolynomialFeatures(2), Ridge())
            model.fit(xm.reshape(-1, 1), ym)
            yr = model.predict(x.reshape(-1, 1))
        else:
            raise KeyError('Unknown linear model: {}'.format(linear_model))

        trend_lines.append(yr)

        if signal_type == 'F-F0':
            detrended_signals.append(signal - yr)
        elif signal_type == 'F/F0':
            detrended_signals.append(signal / yr)
        elif signal_type == 'F-F0/F0':
            # Calculate normalized peak
            # (F - F0) / (F0)
            detrended_signals.append((signal - yr) / yr)
        else:
            raise KeyError('Unknown signal type: {}'.format(signal_type))
    return np.stack(detrended_signals, axis=1), np.stack(trend_lines, axis=1)


def lowpass_filter(signals, fs, order=FILTER_ORDER, cutoff=FILTER_CUTOFF):
    """ Lowpass filter the data

    :param signals:
        A t x k array of k signals with t timepoints
    :param fs:
        The sample rate for the signals
    :param order:
        The order for the butterworth filter
    :param cutoff:
        the cutoff in Hz for the filter
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff <= 0.0 or normal_cutoff >= 1.0:
        if DEBUG_OPTIMIZER:
            print('Cannot filter, got -3dB: {}'.format(normal_cutoff))
            print('Nyquist rate: {}'.format(nyq))
            print('Sample rate (Hz): {}'.format(fs))
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


def find_key_times(time, signal, percents=50, direction='down'):
    """ Find the key times

    :param ndarray time:
        The time vector
    :param ndarray signal:
        The signal vector
    :param list percents:
        A number or list of numbers of times to extract
    :param str direction:
        "up" or "down". Up for a signal expected to be increasing,
        down for the opposite
    :returns:
        A list of times where the signal crosses those thresholds
    """

    sig_max = np.max(signal)
    sig_min = np.min(signal)

    sig_range = sig_max - sig_min

    if isinstance(percents, (int, float)):
        percents = [percents]
    percents = [float(p)/100.0 * sig_range + sig_min
                for p in percents]

    pct_times = []
    for pct in percents:
        if direction == 'up':
            idx = np.nonzero(signal >= pct)[0]
            if len(idx) < 1:
                pct_time = None
            else:
                pct_time = time[idx[0]] - time[0]
        elif direction == 'down':
            idx = np.nonzero(signal <= pct)[0]
            if len(idx) < 1:
                pct_time = None
            else:
                pct_time = time[idx[0]] - time[0]
        else:
            raise KeyError('Unknown direction: {}'.format(direction))
        pct_times.append(pct_time)
    return pct_times


def calc_velocity_stats(time, signal, direction='up', time_scale=TIME_SCALE):
    """ Calculate the velocity statistics

    :param ndarray time:
        The time array in native units
    :param ndarray signal:
        The signal array
    :param str direction:
        One of 'up' or 'down', which direction to calculate stats
    :param float time_scale:
        The scaling factor to convert the time array to seconds
    """
    if time.shape[0] < 3 or signal.shape[0] < 3:
        return np.nan, np.nan, np.nan

    dt = (time[1:] - time[:-1]) / time_scale
    ds = signal[1:] - signal[:-1]

    vel = ds / dt
    if direction == 'up':
        vel = vel[vel > 0]
    elif direction == 'down':
        vel = -vel[vel < 0]
    else:
        raise KeyError('Unknown direction: {}'.format(direction))

    if vel.shape[0] < 3:
        return np.nan, np.nan, np.nan

    if direction == 'up':
        return np.mean(vel), np.std(vel), np.max(vel)
    else:
        return -np.mean(vel), np.std(vel), -np.max(vel)


def calc_stats_around_peak(time, signal, peak_bounds, valley_rel=0.05,
                           time_scale=TIME_SCALE,
                           skip_model_fit=False):
    """ Calculate the stats around a single peak

    :param ndarray time:
        The time vector for the signal
    :param ndarray signal:
        The signal vector
    :param tuple peak_bounds:
        A tuple of start, peak, stop bounds
    :param float valley_rel:
        The relative value of the bottom of a valley
    :param float time_scale:
        The conversion of the time scale to seconds
    :param bool skip_model_fit:
        If True, skip fitting a decay curve model to the data
    :returns:
        The dictionary of stats for the peak
    """

    start_idx, peak_idx, end_idx = peak_bounds

    peak_value = signal[peak_idx]

    before_peak = signal[start_idx:peak_idx+1]
    after_peak = signal[peak_idx:end_idx+1]

    # Work out the indicies of the min point and the 5% threshold
    before_min_index = np.argmin(before_peak)
    after_min_index = np.argmin(after_peak)

    before_min_value = before_peak[before_min_index]
    after_min_value = after_peak[after_min_index]

    before_cutoff = valley_rel * (peak_value - before_min_value) + before_min_value
    after_cutoff = valley_rel * (peak_value - after_min_value) + after_min_value

    before_cutoff = np.max([before_cutoff, before_min_value])
    after_cutoff = np.max([after_cutoff, after_min_value])

    # Find all the indicies below the threshold
    before_peak_locs = np.nonzero(before_peak <= before_cutoff)[0]
    after_peak_locs = np.nonzero(after_peak <= after_cutoff)[0]
    before_peak_locs = np.append(before_peak_locs, before_min_index)
    after_peak_locs = np.append(after_peak_locs, after_min_index)

    peak_start_index = np.max(before_peak_locs)
    peak_start_index += start_idx
    peak_end_index = np.min(after_peak_locs)
    peak_end_index += peak_idx

    # Get stats for total time
    total_wave_time = time[peak_end_index] - time[peak_start_index]

    # Peak height
    peak_height = min([signal[peak_idx] - signal[peak_end_index],
                       signal[peak_idx] - signal[peak_start_index]])

    # If we don't have a peak, don't try to calculate stats
    if peak_height < 1e-9:
        t50_up = t85_up = t90_up = np.nan
        t_peak = np.nan
        t50_down = t85_down = t90_down = np.nan
        mean_vel_up = std_vel_up = max_vel_up = np.nan
        mean_vel_down = std_vel_down = max_vel_down = np.nan
        fit = None
    else:
        # Times for up and down
        t50_up, t85_up, t90_up = find_key_times(
            time[peak_start_index:peak_idx+1],
            signal[peak_start_index:peak_idx+1],
            percents=[50, 85, 90],
            direction='up')

        t_peak = time[peak_idx] - time[peak_start_index]

        t50_down, t85_down, t90_down = find_key_times(
            time[peak_idx:peak_end_index+1],
            signal[peak_idx:peak_end_index+1],
            percents=[50, 15, 10],
            direction='down')

        # Max velocity up and down
        mean_vel_up, std_vel_up, max_vel_up = calc_velocity_stats(
            time[peak_start_index:peak_idx+1],
            signal[peak_start_index:peak_idx+1],
            direction='up',
            time_scale=time_scale)
        mean_vel_down, std_vel_down, max_vel_down = calc_velocity_stats(
            time[peak_idx:peak_end_index+1],
            signal[peak_idx:peak_end_index+1],
            direction='down',
            time_scale=time_scale)

        # Fit some exponential models to the stats
        if skip_model_fit:
            fit = None
        else:
            fit = fit_model(
                time[peak_idx:peak_end_index+1],
                signal[peak_idx:peak_end_index+1],
                time_scale=time_scale)
    data = {
        'peak_value': peak_value,
        'peak_height': peak_height,
        'peak_index': peak_idx,
        'peak_time': time[peak_idx],
        'peak_start_index': peak_start_index,
        'peak_end_index': peak_end_index,
        'total_wave_time': total_wave_time,
        't50_up': t50_up,
        't85_up': t85_up,
        't90_up': t90_up,
        't_peak': t_peak,
        't50_down': t50_down,
        't85_down': t85_down,
        't90_down': t90_down,
        'mean_vel_up': mean_vel_up,
        'max_vel_up': max_vel_up,
        'std_vel_up': std_vel_up,
        'mean_vel_down': mean_vel_down,
        'max_vel_down': max_vel_down,
        'std_vel_down': std_vel_down,
    }
    if fit is not None:
        data.update(fit.get_single_exp_params())
        data.update(fit.get_double_exp_params())
    return data


def calc_signal_stats(time: np.ndarray,
                      signals: np.ndarray,
                      time_scale: float = TIME_SCALE,
                      samples_around_peak: int = SAMPLES_AROUND_PEAK,
                      skip_model_fit: bool = False):
    """ Calculate useful stats for a set of signals

    :param ndarray time:
        The n x 1 time array
    :param ndarray signals:
        The n x m signal array
    :param float time_scale:
        The conversion of time to seconds
    :param int samples_around_peak:
        Minimum number of samples around a peak before the next peak
    :returns:
        The list of valid signal indicies and the list of stats for each peak found in the signal array
    """

    all_peaks = []
    valid_signal_indicies = []

    for i in range(signals.shape[1]):
        signal = signals[:, i]

        # Mask out invalid values
        sigmask = np.isfinite(signal)
        if not np.any(sigmask):
            print('Empty signal in sample: {}'.format(i))
            continue

        # Mask out signals with discontinuities
        offset_st, offset_ed = np.nonzero(sigmask)[0][[0, -1]]
        if not np.all(sigmask[offset_st:offset_ed+1]):
            print('Non-contiguous signal in sample: {}'.format(i))
            continue

        signal_finite = signal[sigmask]
        time_finite = time[sigmask]

        peak_indicies = peak_local_max(signal_finite,
                                       min_distance=samples_around_peak,
                                       indices=True)

        peaks = refine_signal_peaks(time_finite, signal_finite, peak_indicies,
                                    offset=offset_st,
                                    time_scale=time_scale,
                                    skip_model_fit=skip_model_fit)
        all_peaks.append(peaks)
        valid_signal_indicies.append(i)
    return valid_signal_indicies, all_peaks


def refine_signal_peaks(time, signal, peaks,
                        valley_rel=0.05,
                        min_peak_width=1,
                        min_peak_height=0.0,
                        offset=0,
                        time_scale=TIME_SCALE,
                        skip_model_fit=False):
    """ Refine the raw peak indicies for the signal

    :param ndarray time:
        The n x 1 time array
    :param ndarray signal:
        The n x m signal array
    :param list peaks:
        The list of peak indices called by :py:func:`calc_signal_stats`
    :param float valley_rel:
        What relative fraction of the peak to call the valley floor
    :param min_peak_width:
        Minimum width (in samples) of a peak
    :param min_peak_height:
        Minimum height of a peak (in signal intensity)
    :param int offset:
        Starting temporal index for this dataset
    :param float time_scale:
        The scaling factor to convert to seconds
    :returns:
        A new list of peaks, possibly with some maxima combined
    """

    # Augment the peaks with the beginning and end peaks in the signal
    peaks = list(peaks)
    if 0 not in peaks:
        peaks.append(0)
    if signal.shape[0]-1 not in peaks:
        peaks.append(signal.shape[0]-1)
    peaks = list(sorted(int(p) for p in peaks))

    # Try to fuse peaks together while we have multiple peaks to fuse
    while len(peaks) >= 3:
        new_peaks = [0]
        final_peaks = []

        for peak_bounds in zip(peaks[:-2], peaks[1:-1], peaks[2:]):
            mid = peak_bounds[1]
            stats = calc_stats_around_peak(time, signal, peak_bounds,
                                           valley_rel=valley_rel,
                                           time_scale=time_scale,
                                           skip_model_fit=skip_model_fit)

            peak_start_index = stats['peak_start_index']
            peak_end_index = stats['peak_end_index']

            peak_height = stats['peak_height']

            if all([peak_start_index < mid,
                    peak_end_index > mid,
                    peak_end_index - peak_start_index > min_peak_width,
                    peak_height > min_peak_height]):
                new_peaks.append(mid)

            # Handle mask offset for the indicies
            final_stats = {}
            for key in stats:
                if key.endswith('_index'):
                    final_stats[key] = stats[key] + offset
                else:
                    final_stats[key] = stats[key]

            final_peaks.append(final_stats)

        # Once we converge to a fixed point, return
        new_peaks.append(signal.shape[0]-1)
        if new_peaks == peaks:
            return final_peaks
        peaks = new_peaks

    # Failure
    return []


def select_top_stats(stats, min_score=MIN_STATS_SCORE):
    """ Pick the better peaks to use for analysis

    Each peak is scored by peak area = (peak width * peak height)

    If this peak area is less than min_score * max(peak areas), the peak is discarded

    :param list stats:
        The list of stat dictionaries for each signal
    :param float min_score:
        The minimum score to use to keep this peak
    :returns:
        The filtered list of stat dictionaries
    """

    top_stats = []
    for signal_stats in stats:

        if signal_stats == []:
            top_stats.append([])
            continue

        peak_heights = [s['peak_height'] for s in signal_stats]
        peak_widths = [s['total_wave_time'] for s in signal_stats]

        max_peak_height = max(peak_heights)
        max_peak_width = max(peak_widths)

        scores = [ph * pw / max_peak_height / max_peak_width
                  for ph, pw in zip(peak_heights, peak_widths)]

        top_stats.append([stat for stat, score in zip(signal_stats, scores)
                          if score > min_score])
    return top_stats


def add_summary_stats(stats, time_scale=TIME_SCALE):
    """ Summarize the data per-signal

    :param list stats:
        The list of stat dictionaries for each time series
    :param float time_scale:
        The scaling to convert the time into seconds
    :returns:
        A new list of stats combining the summary and individual track stats
    """

    key_stats = [
        'peak_height',
        'total_wave_time',
        't50_up',
        't85_up',
        't90_up',
        't_peak',
        't50_down',
        't85_down',
        't90_down',
        'mean_vel_up',
        'max_vel_up',
        'mean_vel_down',
        'max_vel_down',
        'se_amp',
        'se_tc',
        'de_amp',
        'de_tc1',
        'de_tc2',
    ]

    final_stats = []
    for signal_stats in stats:
        peak_times = np.array([s['peak_time'] for s in signal_stats])

        if peak_times.shape[0] < 2:
            mean_period = np.nan
            std_period = np.nan
            beats_per_minute = np.nan
        else:
            dtau = peak_times[1:] - peak_times[:-1]
            mean_period = np.mean(dtau)
            std_period = np.std(dtau)
            beats_per_minute = 60 * time_scale / mean_period

        final_stat = {
            'signal_stats': signal_stats,
            'mean_period': mean_period,
            'std_period': std_period,
            'beats_per_minute': beats_per_minute,
        }
        for key in key_stats:
            values = [s[key] for s in signal_stats if s.get(key) is not None]
            if not values:
                continue
            final_stat[key] = np.mean(values)
        final_stats.append(final_stat)
    return final_stats


# Per-Data file processing


def filter_datafile(infile, outfile,
                    data_type=DATA_TYPE,
                    signal_type=SIGNAL_TYPE,
                    filter_cutoff=FILTER_CUTOFF,
                    filter_order=FILTER_ORDER,
                    plot_types=None,
                    time_scale=TIME_SCALE,
                    skip_detrend=False,
                    skip_lowpass=False,
                    skip_model_fit=False,
                    samples_around_peak=SAMPLES_AROUND_PEAK,
                    flip_signal=False,
                    level_shift=None,
                    plotfile=None,
                    linear_model=LINEAR_MODEL):
    """ Apply the filtering operation

    :param Path infile:
        The raw input file to filter
    :param Path outfile:
        The filtered output file to write
    :param str data_type:
        Which input data loader to use
    :param str signal_type:
        Which normalization approach to use (F/F0, F-F0, F-F0/F0)
    :param list[str] plot_types:
        The list of different plots to generate for this signal
    :param float filter_cutoff:
        -3dB point for the filter (Hz)
    :param int filter_order:
        The order of the lowpass butterworth filter
    :param bool skip_detrend:
        Skip the linear detrending step
    :param bool skip_lowpass:
        Skip the lowpass filter step
    :param int samples_around_peak:
        The number of samples around each maxima in the signal
    :param float time_scale:
        The conversion factor from samples to seconds
    :param Path plotfile:
        If not None, the path to save the plot file to
    :returns:
        The stats for all the entries in this data file
    """
    try:
        time, raw_signals = DataReader(data_type).read_infile(
            infile, flip_signal=flip_signal, level_shift=level_shift)
    except OSError as err:
        print('Error {} reading "{}"'.format(err, infile))
        if DEBUG_OPTIMIZER:
            traceback.print_exc()
        return None

    dt = np.nanmedian(time[1:] - time[:-1]) / time_scale  # Seconds
    fs = 1.0 / dt

    signals, trend_lines = remove_trend(raw_signals,
                                        signal_type=signal_type,
                                        skip_detrend=skip_detrend,
                                        linear_model=linear_model)
    if not skip_lowpass:
        signals = lowpass_filter(signals, fs=fs, cutoff=filter_cutoff, order=filter_order)

    signal_indicies, stats = calc_signal_stats(time, signals,
                                               time_scale=time_scale,
                                               samples_around_peak=samples_around_peak,
                                               skip_model_fit=skip_model_fit)
    # Only keep valid signals
    raw_signals = raw_signals[:, signal_indicies]
    signals = signals[:, signal_indicies]

    stats = select_top_stats(stats)
    stats = add_summary_stats(stats, time_scale=time_scale)

    plot_signals(infile, time, signals,
                 plot_types=plot_types,
                 raw_signals=raw_signals,
                 trend_lines=trend_lines,
                 stats=stats,
                 signal_type=signal_type,
                 filter_cutoff=filter_cutoff,
                 time_scale=time_scale,
                 plotfile=plotfile)

    save_outfile(outfile, time, signals)

    return stats


def maybe_analyze_datafile(processing_item: AnalysisParams):
    """ Process the datafile in a subprocess

    :param AnalysisParams processing_item:
        The parameters for this item
    :returns:
        The aggregated stats for this item
    """
    datafile = processing_item.datafile
    plotdir = processing_item.plotdir
    outdir = processing_item.outdir

    outfile = outdir / datafile.name
    if plotdir is None:
        plotfile = None
    else:
        plotfile = plotdir / (datafile.stem + processing_item.plot_suffix)
    print('Processing {}'.format(datafile.name))
    try:
        stat = filter_datafile(datafile, outfile,
                               plot_types=processing_item.plot_types,
                               signal_type=processing_item.signal_type,
                               filter_cutoff=processing_item.filter_cutoff,
                               filter_order=processing_item.filter_order,
                               skip_detrend=processing_item.skip_detrend,
                               skip_lowpass=processing_item.skip_lowpass,
                               skip_model_fit=processing_item.skip_model_fit,
                               samples_around_peak=processing_item.samples_around_peak,
                               data_type=processing_item.data_type,
                               flip_signal=processing_item.flip_signal,
                               level_shift=processing_item.level_shift,
                               time_scale=processing_item.time_scale,
                               linear_model=processing_item.linear_model,
                               plotfile=plotfile)
    except Exception:
        print('Errors processing {}'.format(datafile.name))
        traceback.print_exc()
        stat = None
    return datafile, stat
