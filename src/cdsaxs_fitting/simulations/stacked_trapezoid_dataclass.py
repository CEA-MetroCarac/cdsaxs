from dataclasses import dataclass
from typing import Union
import numpy as xp

try:
    import cupy as cp
    xp = cp
except:
    pass




@dataclass(kw_only=True)
class StackedTrapezoidDataClass:
    heights: Union[float, xp.ndarray[float]]
    rangles: xp.ndarray[xp.ndarray[float]]
    langles: xp.ndarray[xp.ndarray[float]]
    y1: xp.ndarray[float]
    bot_cd: xp.ndarray[float]
    dwx: xp.ndarray[float]
    dwz: xp.ndarray[float]
    i0: xp.ndarray[float]
    bkg_cste: xp.ndarray[float]




