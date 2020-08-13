""" Test Utilities

* :py:class:`FileSystemTestCase`: Integration test with a temp directory

"""

# Imports
import unittest
import tempfile
import pathlib

# For testing, add the local directory to the path
THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
assert BASEDIR.is_dir()

# Test Helpers


class FileSystemTestCase(unittest.TestCase):

    def setUp(self):
        self._tempdir_obj = tempfile.TemporaryDirectory()
        self.tempdir = pathlib.Path(self._tempdir_obj.__enter__()).resolve()

    def tearDown(self):
        self.tempdir = None
        self._tempdir_obj.__exit__(None, None, None)
        self._tempdir_obj = None
