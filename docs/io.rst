File Input/Output Tools
=======================

The :py:class:`DataReader` class defines readers for several common calcium and
electrophysiology trace file formats.

Calcium Imaging
---------------

The standard calcium imaging format is a comma separated value file output by Zen
with the following format:

.. code-block:: text

    WeirdTimeColumn,Area1,Mean1,Total1,Max1,Area2,Mean2,...
    "MS",stupid,number,of,commas,...
    1,57,0.1,inf,inf,58,0.2,...

Where the names for each column are arbitrarily defined by Zen. Their order in the
file is used to extract the ROI areas and mean intensities.

This reader is accessed by calling

.. code-block:: bash

    $ ./analyze_ca_data.py --data-type ca ...

Electrophysiology
-----------------

The standard electrophysiology format is a comma separated value file with the following
format:

.. code-block:: text

    Time,Mean1,Mean2,Mean3,...
    1,0.1,0.2,0.3,...
    2,0.2,0.3,0.4,...

This reader is accessed by calling

.. code-block:: bash

    $ ./analyze_ca_data.py --data-type ephys ...

API Reference
-------------

.. automodule:: multilineage_organoid.io
   :members:
   :undoc-members:
