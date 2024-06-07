import copy
import numpy as np
import pandas as pd
import warnings

from .stacked_trapezoid import StackedTrapezoidSimulation, StackedTrapezoidGeometry

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except:
    CUPY_AVAILABLE = False



class OverlaySimulation(StackedTrapezoidSimulation):

    def __init__(self, qys, qzs, from_fitter=False, use_gpu=False, initial_guess=None):
        super().__init__(qys, qzs, from_fitter=from_fitter, use_gpu=use_gpu, initial_guess=initial_guess)

        self.TrapezoidGeometry = OverlayGeometry(xp=self.xp, from_fitter=self.from_fitter, initial_guess=initial_guess)



class OverlayGeometry(StackedTrapezoidGeometry):

    def calculate_ycoords(self, df):
        super().calculate_ycoords(df)
        warnings.warn("calculate_ycoords method needs to be written for OverlayGeometry")

