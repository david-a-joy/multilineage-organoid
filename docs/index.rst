Calcium Trace Analysis for Cardiac and Multilineage Organoids
=============================================================

This package provides the ``analyze_ca_data.py`` script to call peaks and produce
shape analyses for calcium and electrophysiology traces produced by cardiac
and other kinds of beating organoids [1]_.

To analyze the demo data provided with the package, run the following command:

.. code-block:: bash

    $ ./analyze_ca_data.py ./data --data-type ca --plot-type all

To save the results to a specific directory:

.. code-block:: bash

    $ ./analyze_ca_data.py ./data  --data-type ca --plot-type all --outdir ./stats

To save the plots to a specific directory:

.. code-block:: bash

    $ ./analyze_ca_data.py ./data --data-type ca --plot-type all  --plotdir ./plots

Process the data in parallel using 8 cores, saving the annotations to a directory

.. code-block:: bash

    $ ./analyze_ca_data.py ./data --plot-type all --plotdir ./plots --processes 8

It's probably best to only use as many cores as you have on your computer.

Disable detrending and filtering to peak count on raw data:

.. code-block:: bash

    $ ./analyze_ca_data.py /path/to/data/dir --skip-detrend --skip-lowpass

Options to control the detrending, filtering and peak calling are shown in the
command help:

.. code-block:: bash

    $ ./analyze_ca_data.py -h

If you find this package useful, please cite:

.. [1] Silva, A. C. et al. Developmental co-emergence of cardiac and gut tissues modeled by human iPSC-derived organoids. http://biorxiv.org/lookup/doi/10.1101/2020.04.30.071472 (2020) doi:10.1101/2020.04.30.071472.

Individual Modules
------------------

Details of the various signal analysis modules are described below

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   signals
   models
   plotting
   io
   utils

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
