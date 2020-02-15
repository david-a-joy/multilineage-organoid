# Single cell determination of engineered cardiac microtissue structure and function

Analysis code for "???"

## Installing

This script requires Python 3.7 or greater and several additional python packages.
This code has been tested on OS X 10.15 and Ubuntu 18.04, but may work with minor
modification on other systems.

It is recommended to install and test the code in a virtual environment for
maximum reproducibility:

```{bash}
# Create the virtual environment
python3 -m venv ~/cms_env
source ~/cms_env/bin/activate
```

All commands below assume that `python3` and `pip3` refer to the binaries installed in
the virtual environment. Commands are executed from the base of the git repository
unless otherwise specified.

```{bash}
pip install --upgrade pip

# Install the required packages
pip3 install -r requirements.txt

# Build and install all files in the CM microtissue structure toolbox
python3 setup.py install
```

The `cm_microtissue_func` package can also be installed as a python package:

```{bash}
python3 setup.py bdist_wheel
pip3 install dist/cm_microtissue_func-*.whl
```

After installation, the following scripts will be available:

* `analyze_ca_data.py`: Analyze calcium traces and estimate waveform parameters

The scripts can also be used in an `inplace` install, when run locally from the
`scripts` directory.

```{bash}
python3 setup.py build_ext --inplace
cd scripts
```

Where each script is run from the current directory (e.g. `./analyze_ca_data.py`, etc)

## Analyzing trace data





## Testing

The modules defined in `cm_microtissue_func` have a test suite that can be run
using the `pytest` package.

```{bash}
python3 -m pytest tests
```

It is required to first build all extensions using

```{bash}
python3 setup.py build_ext --inplace
```

before running the test suite.

## Documentation

Documentation for the scripts and individual modules can be built using the
`sphinx` package.

```{bash}
cd docs
make html
```

Documentation will then be available under `docs/_build/html/index.html`
