Peak Calling and Signal Analysis
================================

The main signal analysis function :py:func:`filter_datafile` function loads all traces
in a data file, filters and calls peaks, generates plots, then writes the final
stats out.

To analyze a single data file:

.. code-block:: python

    from pathlib import Path
    from multilineage_organoid import filter_datafile

    stats = filter_datafile(
        infile=Path('data/Exp7_d80_MultilineageOrganoid_pacing_1hz.csv'),
        outfile=Path('filtered/Exp7_d80_MultilineageOrganoid_pacing_1hz.csv'),
        plot_types='all',
        plotfile=Path('plots/Exp7_d80_MultilineageOrganoid_pacing_1hz.png'),
    )

This will filter all traces in ``data/Exp7_d80_MultilineageOrganoid_pacing_1hz.csv``
and produce a new set of filtered traces, plots showing the results of filtering
each trace, and a set of summary stats in the ``stats`` object, one for each trace.

API Reference
-------------

.. automodule:: multilineage_organoid.signals
   :members:
   :undoc-members:
