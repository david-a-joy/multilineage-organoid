""" Model fitting for decay times

Classes:

* :py:class:`ExpModelFit`: Implement multiple exponential curve fit models

Functions:

* :py:func:`fit_model`: Fit decay curves to traces

"""


# Imports
import traceback
from typing import Tuple

# 3rd party
import numpy as np

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

# Our own imports
from .consts import TIME_SCALE, DEBUG_OPTIMIZER


# Classes


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

        self.tmean_min = 0.1  # tmean - split point between the two exps
        self.tmean_max = 0.9  # tmean - split point between the two exps
        self.tsigma_min = 0.01  # tsigma - interpolation weights between the two exps
        self.tsigma_max = 0.2  # tsigma - interpolation weights between the two exps

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
        """ Resample the signal onto the [0, 1] domain """

        scaled_time = np.linspace(0, 1, self.signal.shape[0])
        mask = np.logical_and(np.isfinite(self.signal), np.isfinite(scaled_time))
        self._smooth_signal = self.signal[mask]
        self._scaled_time = scaled_time[mask]
        self.se_offset = self.de_offset = np.min(self._smooth_signal)

    def initialize_guesses(self):
        """ Initial guess for the signal

        Fit a least squares regression to the log of the signal
        """
        self.init_offset = np.min(self._smooth_signal)
        log_signal = np.log(self._smooth_signal - np.min(self._smooth_signal) + 1)
        coeffs = np.polyfit(self._scaled_time, log_signal, 1)

        self.init_tc = -coeffs[0]
        self.init_amp = coeffs[1]
        self.amp_max = 1.5*(np.max(self._smooth_signal) - np.min(self._smooth_signal))

        if DEBUG_OPTIMIZER:
            print(f'Initial Tc Guess:     {self.init_tc:0.4f}')
            print(f'Initial Amp Guess:    {self.init_amp:0.4f}')
            print(f'Initial Offset Guess: {self.init_offset:0.4f}')

    def get_single_guesses(self) -> Tuple[float]:
        """ Get the single parameter guesses """
        param_sets = [
            (self.init_amp, self.amp_min, self.amp_max),  # amplitude
            (self.init_tc, self.tc_min, self.tc_max),  # time constant
            (self.init_offset, self.offset_min, self.offset_max),  # offset
        ]
        guesses = []
        for guess, guess_min, guess_max in param_sets:
            if guess_min is not None and guess < guess_min:
                guess = guess_min
            if guess_max is not None and guess > guess_max:
                guess = guess_max
            guesses.append(guess)
        return tuple(guesses)

    def get_double_guesses(self) -> Tuple[float]:
        """ Get the single parameter guesses """

        param_sets = [
            (self.se_amp, self.amp_min, self.amp_max),  # amplitude
            (self.se_offset, self.offset_min, self.offset_max),  # offset
            (self.se_tc*0.5, self.tc_min, self.tc_max),  # time constant - first slower
            (self.se_tc*1.5, self.tc_min, self.tc_max),  # time constant - second faster
            (0.5, self.tmean_min, self.tmean_max),  # tmean - split point between the two exps
            (0.1, self.tsigma_min, self.tsigma_max),  # tsigma - interpolation weights between the two exps
        ]
        guesses = []
        for guess, guess_min, guess_max in param_sets:
            if guess_min is not None and guess < guess_min:
                guess = guess_min
            if guess_max is not None and guess > guess_max:
                guess = guess_max
            guesses.append(guess)
        return tuple(guesses)

    def fit_single_exp(self):
        """ Fit a single exponential curve """
        # Guess at the initial parameters and boundaries
        param_guesses = self.get_single_guesses()
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
            print(f'Single Tc Fit:     {self.se_tc:0.4f}')
            print(f'Single Amp Fit:    {self.se_amp:0.4f}')
            print(f'Single Offset Fit: {self.se_offset:0.4f}')

    def fit_double_exp(self):
        """ Fit a double exponential curve """
        # Guess at the initial parameters and boundaries
        # amp, offset, tc1, tc2, tmean, tsigma
        param_guesses = self.get_double_guesses()
        param_bounds = [
            (self.amp_min, self.offset_min, self.tc_min, self.tc_min, self.tmean_min, self.tsigma_min),
            (self.amp_max, self.offset_max, self.tc_max, self.tc_max, self.tmean_max, self.tsigma_max),
        ]
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
            print(f'Double Amp Fit:    {self.de_amp:0.4f}')
            print(f'Double Offset Fit: {self.de_offset:0.4f}')
            print(f'Double Tc1 Fit:    {self.de_tc1:0.4f}')
            print(f'Double Tc2 Fit:    {self.de_tc2:0.4f}')
            print(f'Double Tmean Fit:  {self.de_tmean:0.4f}')
            print(f'Double Tsigma Fit: {self.de_tsigma:0.4f}')

    def predict_single_exp_model(self) -> np.ndarray:
        """ Predict the expected values for the single exp fit

        :returns:
            The y values for the single exponential model
        """
        return self.single_exp_model(self._scaled_time, self.se_amp, self.se_tc, self.se_offset)

    def predict_double_exp_model(self) -> np.ndarray:
        """ Predict the expected values for the double exp fit

        :returns:
            The y values for the double exponential model
        """
        return self.double_exp_model(self._scaled_time, self.de_amp,
                                     self.de_offset, self.de_tc1, self.de_tc2,
                                     self.de_tmean, self.de_tsigma)

    def plot_fit(self):
        """ Plot the model fit """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))

        se_signal = self.predict_single_exp_model()
        de_signal = self.predict_double_exp_model()

        ax1.plot(self._scaled_time, se_signal, '-r')
        ax1.plot(self._scaled_time, self._smooth_signal, 'om')

        ax2.plot(self._scaled_time, de_signal, '-r')
        ax2.plot(self._scaled_time, self._smooth_signal, 'om')

        fig.suptitle(f'{se_signal.shape[0]} Points')

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
                         tsigma: float) -> np.ndarray:
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
                         offset: float) -> np.ndarray:
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
