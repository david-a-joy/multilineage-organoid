#!/usr/bin/env python3

import pathlib

import setuptools

basedir = pathlib.Path(__file__).resolve().parent
aboutfile = basedir / 'multilineage_organoid' / '__about__.py'
scriptdir = basedir / 'scripts'
readme = basedir / 'README.md'

# Load the info from the about file
about = {}
with aboutfile.open('rt') as fp:
    exec(fp.read(), about)

with readme.open('rt') as fp:
    long_description = fp.read()

scripts = [str(p.relative_to(basedir)) for p in scriptdir.iterdir()
           if not p.name.startswith('.') and p.suffix == '.py' and p.is_file()]

setuptools.setup(
    name=about['__package_name__'],
    version=about['__version__'],
    url=about['__url__'],
    description=about['__description__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=about['__author__'],
    author_email=about['__author_email__'],
    packages=('multilineage_organoid', ),
    scripts=scripts,
)
