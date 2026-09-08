# -*- coding: utf-8 -*-
"""Minimal installer. The suite normally runs from the repository root with
PYTHONPATH set (see the Makefile and the Docker image); this exists so that
``pip install -e .`` also works for interactive use."""
from setuptools import find_packages, setup

setup(
    name="cvloc",
    version="0.1.0",
    description="Cross-validated estimation of the localization radius in "
                "ensemble data assimilation, solved with the Flower "
                "Pollination Algorithm",
    packages=find_packages(include=["cvloc", "cvloc.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26,<3", "scipy>=1.11", "pandas>=2.0",
        "matplotlib>=3.7", "scikit-learn>=1.3",
    ],
)
