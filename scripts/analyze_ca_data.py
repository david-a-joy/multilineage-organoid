#!/usr/bin/env python

""" Signal processing to filter cardiac calcium traces

Process the data and write a stat file:

.. code-block:: bash

    $ ./analyze_ca_data.py /path/to/data/dir

Process the data and plot the annotations on the curves

.. code-block:: bash

    $ ./analyze_ca_data.py /path/to/data/dir --plot-type all

Process the data, plot the annotations, and save them to a directory

.. code-block:: bash

    $ ./analyze_ca_data.py /path/to/data/dir --plot-type all --plotdir /path/to/plot/dir

Process the data, plot the annotations, and save them to a directory as SVGs for editing in illustrator

.. code-block:: bash

    $ ./analyze_ca_data.py /path/to/data/dir --plot-type all --plotdir /path/to/plot/dir --plot-suffix '.svg'

Process the data in parallel using 8 cores, saving the annotations to a directory

.. code-block:: bash

    $ ./analyze_ca_data.py /path/to/data/dir --plot-type all --plotdir /path/to/plot/dir --processes 8

It's probably best to only use as many cores as you have on your computer.

Process the data and show only the filtered data

.. code-block:: bash

    $ ./analyze_ca_data.py /path/to/data/dir --plot-type filt

Process the data and show only the filtered and raw data

.. code-block:: bash

    $ ./analyze_ca_data.py /path/to/data/dir --plot-type filt --plot-type raw

Process the data and show only the frequency spectrum and raw data

.. code-block:: bash

    $ ./analyze_ca_data.py /path/to/data/dir --plot-type freq --plot-type raw

Disable detrending and filtering to peak count on raw data:

.. code-block:: bash

    $ ./analyze_ca_data.py /path/to/data/dir --skip-detrend --skip-lowpass

Options to control the detrending, filtering and peak calling are shown in the
command help:

.. code-block:: bash

    $ ./analyze_ca_data.py -h

"""

# Imports
import sys
import shutil
import pathlib
import argparse
import warnings
import textwrap
import multiprocessing
from typing import Optional, List

# For testing, add the local directory to the path
THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if THISDIR.name == 'scripts' and (BASEDIR / 'cm_microtissue_toolbox').is_dir():
    sys.path.insert(0, str(BASEDIR))

# Our own imports
from cm_microtissue_func import signals
from cm_microtissue_func.io import save_final_stats
from cm_microtissue_func.consts import (
    SIGNAL_TYPE, FILTER_CUTOFF, SAMPLES_AROUND_PEAK, LINEAR_MODEL, DATA_TYPE,
    TIME_SCALE, PLOT_SUFFIX, FILTER_ORDER
)

# Constants

if not signals.DEBUG_OPTIMIZER:
    warnings.simplefilter('ignore')

# Main Function


def analyze_ca_data(datadir: pathlib.Path,
                    plot_types: Optional[List[str]] = None,
                    statfile: Optional[pathlib.Path] = None,
                    outdir: Optional[pathlib.Path] = None,
                    plotdir: Optional[pathlib.Path] = None,
                    processes: Optional[int] = None,
                    **kwargs):
    """ Analyze all the data in the folder

    :param Path datadir:
        The directory containing a set of '.csv' files to analyze
    :param list[str] plot_types:
        The list of debugging plots to render ('all' for all, 'none' to show no plots)
    :param Path statfile:
        The path to write the final stats out to
    :param Path plotdir:
        The path to save plots to
    :param str plot_suffix:
        The suffix to save plots with (either '.png' or '.svg')
    :param int processes:
        The number of parallel processes to run the filtering with
    """
    # Work out which plot(s) we got requested
    if plot_types in (None, []):
        plot_types = ['none']
    elif isinstance(plot_types, str):
        plot_types = [plot_types]
    elif 'none' in plot_types:
        plot_types = []
    if 'all' in plot_types:
        plot_types = ['raw', 'filtered', 'freq']
    if 'filt' in plot_types or 'processed' in plot_types:
        plot_types.append('filtered')

    # Work out the path to the filtered data
    if outdir is None:
        if datadir.is_file():
            outdir = datadir.parent / 'filtered'
        else:
            outdir = datadir / 'filtered'
    if outdir.is_dir():
        print(f'Clearing old output: {outdir}')
        shutil.rmtree(str(outdir))
    outdir.mkdir(parents=True)

    if statfile is None:
        statfile = outdir / 'stats.xlsx'

    if processes is None:
        processes = 1
    if processes > 1 and plotdir is None and plot_types != []:
        raise ValueError('Cannot use parallel processing without a plot directory')

    stats = {}
    processing_items = (signals.AnalysisParams(datafile=datafile,
                                               plot_types=plot_types,
                                               plotdir=plotdir,
                                               outdir=outdir,
                                               **kwargs)
                        for datafile in signals.find_ca_data(datadir))

    # Do parallel processing on the stats
    if processes <= 1:
        map_fxn = map
    else:
        pool = multiprocessing.Pool(processes=processes)
        map_fxn = pool.map

    for datafile, stat in map_fxn(signals.maybe_analyze_datafile, processing_items):
        if stat is not None:
            stats[datafile.name] = stat

    save_final_stats(statfile, stats)


# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=pathlib.Path,
                        help='Path to the directory containing raw data (e.g. /path/to/my/data)')
    parser.add_argument('--statfile', type=pathlib.Path,
                        help='Final statistics output file path (e.g. /path/to/stats.xlsx)')
    parser.add_argument('--outdir', type=pathlib.Path,
                        help='Directory to write individual filtered traces to (e.g. /path/to/filtered)')
    parser.add_argument('-p', '--plot-type', dest='plot_types', action='append', default=[],
                        choices=('none', 'raw', 'processed', 'filt', 'filtered', 'freq', 'all'),
                        help='Which kinds of plots to generate (raw, processed, frequency domain, etc)')
    parser.add_argument('--signal-type', default=SIGNAL_TYPE,
                        choices=('F/F0', 'F-F0/F0', 'F-F0'),
                        help='Method used to normalize the signal to baseline')
    parser.add_argument('--filter-cutoff', type=float, default=FILTER_CUTOFF,
                        help='-3dB cutoff for the filter (Hz)')
    parser.add_argument('--filter-order', type=int, default=FILTER_ORDER,
                        help='Order of the butterworth low-pass filter')
    parser.add_argument('--skip-detrend', action='store_true',
                        help='Skip the detrending step of the processing')
    parser.add_argument('--skip-lowpass', action='store_true',
                        help='Skip the lowpass filter step of the processing')
    parser.add_argument('--skip-model-fit', action='store_true',
                        help='Skip model fit on the decay curve estimation step of the processing')
    parser.add_argument('--samples-around-peak', type=int, default=SAMPLES_AROUND_PEAK,
                        help='Minimum number of samples around each peak')
    parser.add_argument('--data-type', choices=tuple(signals.DataReader.get_readers()), default=DATA_TYPE,
                        help='Which type of input data to load')
    parser.add_argument('--flip-signal', action='store_true',
                        help='Flip the signal upside down')
    parser.add_argument('--level-shift', choices=('min', 'range', 'one', 'zero'),
                        help=textwrap.dedent('''
                            How to shift the raw data when doing F/F0 normalization.
                            Min sets F0 at the abs(min).
                            Range sets F0 at abs(max - min).
                            One sets F0 === 1.0 (which may break things)...
                            Zero sets F0 === 0.0 (which may break things)...
                        '''))
    parser.add_argument('--time-scale', type=float, default=TIME_SCALE,
                        help='Conversion factor to scale to seconds')
    parser.add_argument('--plotdir', type=pathlib.Path,
                        help='Path to save the plots to')
    parser.add_argument('--plot-suffix', type=str, choices=('.png', '.svg'), default=PLOT_SUFFIX,
                        help='Suffix to save the plots with')
    parser.add_argument('--processes', type=int,
                        help='Run the filtering with this many parallel processes')
    parser.add_argument('--linear-model', choices=('linear', 'ransac', 'exp_ransac', 'quadratic', 'boxcar'),
                        default=LINEAR_MODEL,
                        help='Which model to fit for signal detrending')
    return parser.parse_args(args=args)


def main(args=None):
    args = vars(parse_args(args=args))
    analyze_ca_data(**args)


if __name__ == '__main__':
    main()
