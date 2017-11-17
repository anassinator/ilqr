# -*- coding: utf-8 -*-
"""Iterative Linear Quadratic Regulator."""

__version__ = "0.1.0"

from .controller import iLQR, RecedingHorizonController

__all__ = ["iLQR", "RecedingHorizonController"]
