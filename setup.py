# -*- coding: utf-8 -*-
"""
    Setup file for textsumm.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import sys

from pkg_resources import VersionConflict, require
from setuptools import setup, find_packages

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)

with open("README.md", 'r') as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(
    name="TextSumm",
    version="0.0.1",
    author="Satya Siddharth Dash",
    author_email="dash.sathyasiddharth@gmail.com",
    description="Text Summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SatyaSiddharthDash/TextSumm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
