#!/usr/bin/env python

import os
from ilqr import __version__
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


base_url = "https://github.com/anassinator/ilqr"

setup(
    name="ilqr",
    version=__version__,
    description="Auto-differentiated Iterative Linear Quadratic Regulator",
    long_description=read("README.rst"),
    author="Anass Al",
    author_email="mr@anassinator.com",
    license="GPLv3",
    url=base_url,
    download_url="{}/tarball/{}".format(base_url, __version__),
    packages=["ilqr", "ilqr.examples"],
    zip_safe=True,
    install_requires=[read("requirements.txt").strip().split("\n")],
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
