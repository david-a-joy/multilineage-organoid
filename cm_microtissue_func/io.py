""" Input/Output helpers

Classes:

* :py:class:`DataReader`: Read a number of imaging and ephys file formats

Functions:

* :py:func:`find_ca_data`: Find trace data under a directory
* :py:func:`save_outfile`: Write out the individual processed traces
* :py:func:`save_final_stats`: Write out the summary stats for all the traces

"""

# Imports
import re
import inspect
import pathlib
from typing import Dict, Generator, Any

# 3rd party
import numpy as np

import pandas as pd

# Our own imports
from .consts import DATA_TYPE

# Column Name Parsers

TIME_KEY1 = re.compile(r'''^
    time\:\:.*time\!\!r
$''', re.IGNORECASE | re.VERBOSE)
AREA_KEY1 = re.compile(r'''^
    channel[0-9]+_R[0-9]+_area\:\:.*area\!\!r
$''', re.IGNORECASE | re.VERBOSE)
MEAN_KEY1 = re.compile(r'''^
    channel[0-9]+_R[0-9]+_intensitymean\:\:.*intensitymean\!\!r
$''', re.IGNORECASE | re.VERBOSE)

TIME_KEYS = [
    TIME_KEY1,
    re.compile(r'^time$', re.IGNORECASE),
    re.compile(r'^.*t\:\:time\!\!.*$', re.IGNORECASE),
]
AREA_KEYS = [
    AREA_KEY1,
    re.compile(r'^area\s*[0-9]+$', re.IGNORECASE),
    re.compile(r'^.*area\:\:area\!\!.*$', re.IGNORECASE),
]
MEAN_KEYS = [
    MEAN_KEY1,
    re.compile(r'^mean\s*[0-9]+$', re.IGNORECASE),
    re.compile(r'^.*value\:\:intensity.*$', re.IGNORECASE),
]
NAME_KEYS = [
    re.compile(r'^.*name\:\:name.*$', re.IGNORECASE),
]

# File Name Patterns

READER_NAME = re.compile(r'^read_infile_(?P<name>[a-z0-9]+)_data$', re.IGNORECASE)

SUFFIXES = ('.csv', '.tsv', '.xls', '.xlsx')


# Classes


class DataReader(object):
    """ Read data from different data types

    Readers for different file types:

    * :py:meth:`read_infile_ca_data`: Data from Zen GCAMP imaging
    * :py:meth:`read_infile_ephys_data`: Data from electrophysiology

    Define a new reader:

    .. code-block:: python

        def read_infile_mytype_data(self, infile, **kwargs):
            # read the file here

    To use call:

    .. code-block:: bash

        $ analyze_ca_data.py --data-type mytype /path/to/data

    """

    def __init__(self, data_type: str = DATA_TYPE):
        self.data_type = data_type

    @classmethod
    def get_readers(cls):
        """ Get the supported readers for this class

        :returns:
            A dictionary of reader_name: reader_method pairs
        """
        readers = {}
        for meth_name, meth in inspect.getmembers(cls(), inspect.ismethod):
            match = READER_NAME.match(meth_name)
            if not match:
                continue
            readers[match.group('name')] = meth
        return readers

    def read_data_frame(self, infile: pathlib.Path, **kwargs):
        """ Read in a data frame from a spreadsheet

        :param Path infile:
            The input file to read
        :returns:
            A DataFrame containing the contents of the file
        """
        infile = pathlib.Path(infile)
        if not infile.is_file():
            raise OSError(f'Cannot find file "{infile}"')

        # Read in the file based on file suffix
        if infile.name.endswith('.csv'):
            return pd.read_csv(str(infile), sep=',', **kwargs)
        elif infile.name.endswith('.tsv'):
            return pd.read_csv(str(infile), sep='\t', **kwargs)
        elif infile.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(str(infile), **kwargs)
        else:
            raise KeyError(f'No DataFrame reader for file type "{infile}"')

    def read_infile(self, infile: pathlib.Path, **kwargs):
        """ Dispatch based on type

        :param Path infile:
            The input file to read
        :param \\*\\* kwargs:
            The arguments to the individual readers
        """
        flip_signal = kwargs.pop('flip_signal', False)
        level_shift = kwargs.pop('level_shift', None)

        # Tricky method lookup dispatch
        # Call a method named ``read_infile_{data_type}`` and return the results
        time, means = self.get_readers()[self.data_type](infile, **kwargs)

        # Calculate the min/max of each signal
        min_means = np.nanmin(means, axis=0)
        max_means = np.nanmax(means, axis=0)
        rng_means = (max_means - min_means)

        if flip_signal:
            # Flip the signal without level shifting
            means = (means - min_means) / rng_means
            means = (1.0 - means) * rng_means + min_means

        # Level shift all the means to prevent stupid things happening around 0
        level_min = 1.0
        if level_shift in (None, 'min'):
            shift = np.abs(np.nanmin(means, axis=0))
        elif level_shift == 'range':
            shift = np.abs(rng_means.copy())
        elif level_shift == 'one':
            shift = np.ones_like(rng_means)
        elif level_shift == 'zero':
            shift = np.zeros_like(rng_means)
            level_min = 0.0  # hope you know what you're doing...
        else:
            raise ValueError(f'Unknown level shift setting: {level_shift}')

        # After this point, the signal should be everywhere positive and >= level_min
        shift[shift < level_min] = level_min
        means = means - np.nanmin(means, axis=0) + shift

        print('Got {} traces with {} samples each'.format(means.shape[1], means.shape[0]))
        assert means.shape[0] == time.shape[0]
        return time, means

    def read_infile_ephys_data(self, infile: pathlib.Path):
        """ Read Mike's data

        Ephys data is a CSV file with or without a header with columns:

        .. code-block:: text

            Time,Mean1,Mean2,Mean3,...
            1,0.1,0.2,0.3,...
            2,0.2,0.3,0.4,...

        All means are assumed aligned in time with no filtering for invalid values.

        Most of these types of files only have one mean/file, which makes it easier.

        :param Path infile:
            The input '.csv' files to read
        :returns:
            (time, means) for each roi in the file
        """
        # Try the first time without a header
        df = self.read_data_frame(infile, header=None)
        time = df.iloc[:, 0]
        if time.dtype not in (np.dtype('float64'), np.dtype('int64')):
            df = self.read_data_frame(infile, header=None, skiprows=1)
        time = df.iloc[:, 0]
        if time.dtype not in (np.dtype('float64'), np.dtype('int64')):
            raise ValueError('Cannot understand file header: {}'.format(infile))
        time = df.iloc[:, 0].values
        means = df.iloc[:, 1:].values

        assert time.shape[0] == means.shape[0]
        return time, means

    def read_infile_ca_data(self,
                            infile: pathlib.Path,
                            min_samples: int = 10):
        """ Read the Ca data input file

        CA GCAMP imaging has a crazy file structure:

        .. code-block:: bash

            WeirdTimeColumn,Area1,Mean1,Total1,Max1,Area2,Mean2,...
            "MS",stupid,number,of,commas,...
            1,57,0.1,inf,inf,58,0.2,...

        If the means are not aligned in time, the time or area values are negative,
        this reader masks those values out as NaNs.

        This loader also tries to filter out super short reads, because they seem to
        be errors too...

        :param Path infile:
            The input '.csv' file to read
        :param int min_samples:
            Minimum number of samples to allow in a time series
        :returns:
            (time, means) for each roi in the file
        """

        time_indices = []
        area_indices = []
        mean_indices = []

        df = self.read_data_frame(infile)
        df = df.iloc[1:, :]

        column_matchers = [
            (TIME_KEYS, time_indices),
            (AREA_KEYS, area_indices),
            (MEAN_KEYS, mean_indices),
        ]

        for colidx, column in enumerate(df.columns):
            # Work out column indices by matching on the name
            for colkeys, colindices in column_matchers:
                had_match = False
                for colkey in colkeys:
                    if colkey.match(str(column).strip()):
                        colindices.append(colidx)
                        had_match = True
                        break
                if had_match:
                    break

        # Sanity checks for badly formatted files
        if len(time_indices) == 0:
            err = f'Invalid stat file. Cannot find the time index in: {infile}'
            raise OSError(err)
        elif len(time_indices) > 1:
            err = 'Invalid stat file. Columns {} all appear to be time records in: {}'
            err = err.format(time_indices, infile)
            raise OSError(err)
        time_idx = time_indices[0]

        if len(area_indices) == 0:
            err = f'Invalid stat file. Cannot find any area columns in: {infile}'
            raise OSError(err)
        if len(mean_indices) == 0:
            err = f'Invalid stat file. Cannot find any signal value columns in: {infile}'
            raise OSError(err)

        if len(area_indices) != len(mean_indices):
            err = 'Invalid stat file. Got {} unique ROIs but {} signal values in: {}'
            err = err.format(len(area_indices), len(mean_indices), infile)
            raise OSError(err)

        # Make sure we're extracting the columns properly
        # print('Time column:  {}'.format(time_idx))
        # print('Area columns: {}'.format(area_indices))
        # print('Mean columns: {}'.format(mean_indices))

        time = df.iloc[:, time_idx].values.astype(np.float)
        areas = df.iloc[:, area_indices].values
        means = df.iloc[:, mean_indices].values

        # Area == 0 means that the ROI isn't valid for this timepoint
        means[areas <= 0] = np.nan

        # Time < 0 means that this sample isn't valid for this timepoint
        means[time < 0] = np.nan

        keep_cols = np.sum(~np.isnan(means), axis=0) >= min_samples
        means = means[:, keep_cols]
        if means.shape[1] < 1:
            err = f'Empty stat file. No ROIs with at least {min_samples} samples in {infile}'
            raise OSError(err)

        return time, means

    def read_infile_measure_data(self,
                                 infile: pathlib.Path,
                                 min_samples: int = 10):
        """ Read the Zen Measure Tool Output file

        CA GCAMP imaging has a crazy file structure:

        .. code-block:: bash

            WeirdNameColumn,WeirdTime,WeirdArea,WeirdDiameter
            "MS",stupid,number,of,commas,...
            name1,time1,area1,diameter1,
            name2,time1,area1,diameter1,
            name1,time2,area2,diameter2,
            name2,time2,area2,diameter2,

        If the means are not aligned in time, the time or area values are negative,
        this reader masks those values out as NaNs.

        This loader also tries to filter out super short reads, because they seem to
        be errors too...

        :param Path infile:
            The input '.csv' file to read
        :param int min_samples:
            Minimum number of samples to allow in a time series
        :returns:
            (time, means) for each roi in the file
        """

        name_indices = []
        time_indices = []
        area_indices = []
        mean_indices = []

        df = self.read_data_frame(infile)
        df = df.iloc[1:, :]

        column_matchers = [
            (NAME_KEYS, name_indices),
            (TIME_KEYS, time_indices),
            (AREA_KEYS, area_indices),
            (MEAN_KEYS, mean_indices),
        ]

        for colidx, column in enumerate(df.columns):
            # Work out column indices by matching on the name
            for colkeys, colindices in column_matchers:
                had_match = False
                for colkey in colkeys:
                    if colkey.match(str(column).strip()):
                        colindices.append(colidx)
                        had_match = True
                        break
                if had_match:
                    break

        # Sanity checks for badly formatted files
        if len(time_indices) == 0:
            err = 'Invalid stat file. Cannot find the time index in: {}'
            err = err.format(infile)
            raise OSError(err)
        elif len(time_indices) > 1:
            err = 'Invalid stat file. Columns {} all appear to be time records in: {}'
            err = err.format(time_indices, infile)
            raise OSError(err)
        time_idx = time_indices[0]

        if len(area_indices) == 0:
            err = 'Invalid stat file. Cannot find any area columns in: {}'
            err = err.format(infile)
            raise OSError(err)
        elif len(area_indices) > 1:
            err = 'Invalid stat file. Columns {} all appear to be area records in: {}'
            err = err.format(area_indices, infile)
            raise OSError(err)
        area_idx = area_indices[0]

        if len(mean_indices) == 0:
            err = 'Invalid stat file. Cannot find any signal value columns in: {}'
            err = err.format(infile)
            raise OSError(err)
        elif len(mean_indices) > 1:
            err = 'Invalid stat file. Columns {} all appear to be signal value records in: {}'
            err = err.format(mean_indices, infile)
            raise OSError(err)
        mean_idx = mean_indices[0]

        if len(name_indices) == 0:
            err = 'Invalid stat file. Cannot find any name columns in: {}'
            err = err.format(infile)
            raise OSError(err)
        elif len(name_indices) > 1:
            err = 'Invalid stat file. Columns {} all appear to be name records in: {}'
            err = err.format(mean_indices, infile)
            raise OSError(err)
        name_idx = name_indices[0]

        # Make sure we're extracting the columns properly
        # print('Time column:  {}'.format(time_idx))
        # print('Name column: {}'.format(name_idx))
        # print('Area column: {}'.format(area_idx))
        # print('Mean column: {}'.format(mean_idx))

        names = df.iloc[:, name_idx].values
        times = df.iloc[:, time_idx].values.astype(np.float)
        areas = df.iloc[:, area_idx].values.astype(np.float)
        means = df.iloc[:, mean_idx].values.astype(np.float)

        # FIXME: Handle mis-aligned times here
        unique_names = np.unique(names)
        stack_times = None
        stack_areas = []
        stack_means = []
        for name in unique_names:
            mask = names == name
            if stack_times is None:
                stack_times = times[mask]
            else:
                assert np.allclose(stack_times, times[mask])
            stack_areas.append(areas[mask])
            stack_means.append(means[mask])

        times = stack_times
        means = np.stack(stack_means, axis=1)
        areas = np.stack(stack_areas, axis=1)

        # Area == 0 means that the ROI isn't valid for this timepoint
        means[areas <= 0] = np.nan

        # Time < 0 means that this sample isn't valid for this timepoint
        means[times < 0] = np.nan

        keep_cols = np.sum(~np.isnan(means), axis=0) >= min_samples
        means = means[:, keep_cols]
        if means.shape[1] < 1:
            err = 'Empty stat file. No ROIs with at least {} samples in {}'
            err = err.format(min_samples, infile)
            raise OSError(err)

        return times, means

# Functions


def save_outfile(outfile: pathlib.Path,
                 time: np.ndarray,
                 signals: np.ndarray):
    """ Save the filtered output

    :param Path outfile:
        The path to the filtered output CSV file
    :param ndarray time:
        The n x 1 array of time steps to save
    :param ndarray signals:
        The n x m array of signals to save
    """

    columns = ['Time']

    df = {'Time': time}
    for i in range(signals.shape[1]):
        key = f'Mean{i + 1:d}'
        columns.append(key)
        df[key] = signals[:, i]

    df = pd.DataFrame(df)
    df.to_csv(str(outfile), columns=columns, index=False)


def save_final_stats(outfile: pathlib.Path,
                     stats: Dict[str, Dict[str, Any]]):
    """ Save the composite stats for everything

    :param Path outfile:
        The path to the stat file to save
    :param dict stats:
        The dictionary of stats to save
    """

    df = {}
    for key, stat in sorted(stats.items()):
        if key == 'signal_stats':
            continue
        for i, rec in enumerate(stat):
            df.setdefault('Filename', []).append(key)
            df.setdefault('Trace', []).append(i + 1)
            for k, v in rec.items():
                df.setdefault(k, []).append(v)

    assert all(len(v) == len(df['Filename']) for v in df.values())

    print('Writing: {}'.format(outfile))
    df = pd.DataFrame(df)
    df.to_excel(str(outfile), index=False)


def find_ca_data(datadir: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    """ Find all the CA data in the data directory

    :param datadir:
        The data directory to search
    :returns:
        An iterator yielding paths to CA files
    """
    if datadir.is_file() and datadir.suffix in SUFFIXES:
        yield datadir
    else:
        for datafile in datadir.iterdir():
            if datafile.name.startswith('.'):
                continue
            if not datafile.is_file():
                continue
            # Ignore the outputs from the old analysis code
            if datafile.name == 'stats.xlsx':
                continue
            if datafile.suffix in SUFFIXES:
                yield datafile
