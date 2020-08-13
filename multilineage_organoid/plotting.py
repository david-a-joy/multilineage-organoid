""" Plotting tools for individual traces

Functions:

* :py:func:`plot_signals`: Plot the traces for the signal sets

"""

# Imports
import pathlib
from typing import Optional, List, Dict, Tuple

# 3rd party
import numpy as np

import matplotlib.pyplot as plt

# Our own imports
from .consts import SIGNAL_TYPE, TIME_SCALE, FILTER_CUTOFF, FIGSIZE, DEBUG_OPTIMIZER
from .models import ExpModelFit
from .utils import calc_frequency_domain

# Functions


def plot_signals(infile: pathlib.Path,
                 time: np.ndarray,
                 signals: np.ndarray,
                 plot_types: Optional[List[str]] = None,
                 raw_signals: Optional[np.ndarray] = None,
                 trend_lines: Optional[np.ndarray] = None,
                 stats:  List[Dict] = None,
                 signal_type: str = SIGNAL_TYPE,
                 time_scale: float = TIME_SCALE,
                 filter_cutoff: float = FILTER_CUTOFF,
                 figsize: Tuple[float] = FIGSIZE,
                 plotfile: Optional[pathlib.Path] = None,
                 single_model_color: str = '#00FF00',
                 double_model_color: str = '#009900'):
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
        err = f'Got {len(stats)} stat measurements but {signals.shape[1]} signals'
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
            raise KeyError(f'Unknown signal type: "{signal_type}"')
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
                raise KeyError(f'Unknown signal type: "{signal_type}"')
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
            labeled_single = False
            labeled_double = False
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

                # Skip peak detections that failed
                if (peak_end_index - peak_index) < 3 or peak_range < 1e-5:
                    continue
                scaled_peak_time = (peak_time - peak_time_offset)/peak_range

                # Single exponential fit
                if 'se_amp' in peak_stats:
                    scaled_peak_fit = ExpModelFit.single_exp_model(scaled_peak_time,
                                                                   peak_stats['se_amp'],
                                                                   peak_range/peak_stats['se_tc'],
                                                                   peak_stats['se_offset'])
                    if np.any(np.isfinite(scaled_peak_fit)):
                        if not labeled_single:
                            labeled_single = True
                            label = 'single exp'
                        else:
                            label = None
                        ax0.plot(peak_time, scaled_peak_fit, '-',
                                 color=single_model_color, label=label)
                    elif DEBUG_OPTIMIZER:
                        print(f'Invalid single fit for {infile.stem}')

                # Double exponential fit
                if 'de_amp' in peak_stats:
                    scaled_peak_fit = ExpModelFit.double_exp_model(
                        scaled_peak_time, peak_stats['de_amp'], peak_stats['de_offset'],
                        peak_range/peak_stats['de_tc1'], peak_range/peak_stats['de_tc2'],
                        peak_stats['de_tmean'], peak_stats['de_tsigma'],
                    )
                    if np.any(np.isfinite(scaled_peak_fit)):
                        if not labeled_double:
                            labeled_double = True
                            label = 'double exp'
                        else:
                            label = None
                        ax0.plot(peak_time, scaled_peak_fit, '-',
                                 color=double_model_color, label=label)
                    elif DEBUG_OPTIMIZER:
                        print(f'Invalid double fit for {infile.stem}')

        if 'raw' in plot_types or 'filtered' in plot_types:
            ax0.set_xlabel('Time (ms)')
            ax0.set_ylabel('Intensity')
            ax0.set_title(f'{signal_type} Intensity {infile.stem}')
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
            ax1.set_title(f'{signal_type} Intensity {infile.stem}')
            ax1.legend()

        if plotfile is None:
            plt.show()
        else:
            plotfile = pathlib.Path(plotfile)
            final_plotfile = plotfile.parent / f'{plotfile.stem}_{i+1:02d}{plotfile.suffix}'
            final_plotfile.parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(str(final_plotfile))
            plt.close()
