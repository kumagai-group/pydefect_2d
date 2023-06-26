# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from math import erf, exp
from typing import List

import numpy as np

from pydefect_2d.potential.grids import Grid


def make_step_like_distribution(grid: Grid,
                                step_left: float,
                                step_right: float,
                                error_func_width) -> np.ndarray:
    """ Make step-like distribution

    :param grid: Cartesian coordinates in Å.
    :param step_left: Cartesian coord in Å
    :param step_right: Cartesian coord in Å
    :param error_func_width: Width in Å.

    """

    def func_left(dist):
        return - erf(dist / error_func_width) / 2 + 0.5

    def func_right(dist):
        return erf(dist / error_func_width) / 2 + 0.5

    result = []
    for g in grid.grid_points:
        d = {"l": step_left - g,
             "l_p1": step_left - g + grid.length,
             "l_m1": step_left - g - grid.length,
             "r": step_right - g,
             "r_p1": step_right - g + grid.length,
             "r_m1": step_right - g - grid.length}
        dd = {k: abs(v) for k, v in d.items()}
        shortest = min(d, key=dd.get)

        if shortest[0] == "l":
            result.append(func_left(d[shortest]))
        else:
            result.append(func_right(d[shortest]))

    return np.array(result)


def make_gaussian_distribution(grid: Grid,
                               position: float,
                               sigma: float) -> np.array:
    """Make gaussian dist. w/o normalization under periodic boundary condition.

    All lengths are in Å.
    """
    def gaussian(length):
        return exp(-length**2/(2*sigma**2))

    result = []
    for g in grid.grid_points:
        rel = g - position
        shortest = min([abs(rel), abs(rel - grid.length), abs(rel + grid.length)])
        result.append(gaussian(shortest))

    return np.array(result)


def rescale_distribution(dist: np.ndarray, average: float) -> np.array:
    scale = average / np.mean(dist)
    return (np.round(dist * scale, decimals=5))
