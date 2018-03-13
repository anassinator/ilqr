#!/usr/bin/env python
"""iLQR setup."""

import os
from setuptools import setup

from ilqr import __version__


def read(fname):
    """Reads a file's contents as a string.

    Args:
        fname: Filename.

    Returns:
        File's contents.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


BASE_URL = "https://github.com/anassinator/ilqr"
INSTALL_REQUIRES = [
    "numpy==1.14.2",
    "scipy==1.0.0",
    "six==1.11.0",
    "Theano==1.0.1",
]

setup(
    name="ilqr",
    version=__version__,
    description="Auto-differentiated Iterative Linear Quadratic Regulator",
    long_description=read("README.rst"),
    author="Anass Al",
    author_email="mr@anassinator.com",
    license="GPLv3",
    url=BASE_URL,
    download_url="{}/tarball/{}".format(BASE_URL, __version__),
    packages=["ilqr", "ilqr.examples"],
    zip_safe=True,
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ])
