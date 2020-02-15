""" Shared default parameters across the modules """

SIGNAL_TYPE = 'F/F0'  # F-F0, F/F0, F-F0/F0

LINEAR_MODEL = 'ransac'  # 'least_squares' or 'ransac' or 'exp_ransac'

DATA_TYPE = 'ca'  # one of 'ca', 'ephys'

FILTER_ORDER = 1  # Order of the butterworth filter
FILTER_CUTOFF = 4  # Hz cutoff for the filter
MIN_STATS_SCORE = 0.2  # Cutoff for peaks to be sufficiently wavy

FIGSIZE = 8  # inches - the width of a (square) figure panel

PLOT_SUFFIX = '.png'

SAMPLES_AROUND_PEAK = 25  # Minimum number of samples around a peak before the next peak can start

TIME_SCALE = 1000  # milliseconds / second

DEBUG_OPTIMIZER = False  # If True, print optimizer debug messages
