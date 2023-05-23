# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from math import erf, exp
from typing import List

import numpy as np


def make_step_like_distribution(grid: List[float],
                                step_left: float,
                                step_right: float,
                                error_func_width) -> List[float]:
    """ Make step-like distribution

    :param grid: Cartesian coordinates in Å.
    :param step_left: Cartesian coord in Å
    :param step_right: Cartesian coord in Å
    :param error_func_width: Width in Å.

    """
    z_length = grid[-1] + grid[1]

    def func_left(dist):
        return - erf(dist / error_func_width) / 2 + 0.5

    def func_right(dist):
        return erf(dist / error_func_width) / 2 + 0.5

    result = []
    for g in grid:
        d = {"l": step_left - g,
             "l_p1": step_left - g + z_length,
             "l_m1": step_left - g - z_length,
             "r": step_right - g,
             "r_p1": step_right - g + z_length,
             "r_m1": step_right - g - z_length}
        dd = {k: abs(v) for k, v in d.items()}
        shortest = min(d, key=dd.get)

        if shortest[0] == "l":
            result.append(func_left(d[shortest]))
        else:
            result.append(func_right(d[shortest]))

    return result


def make_gaussian_distribution(grid: List[float],
                               position: float,
                               sigma: float) -> List[float]:
    """Make gaussian distribution w/o normalization. """
    z_length = grid[-1] + grid[1]

    def gaussian(length):
        return exp(-length**2/(2*sigma**2))

    result = []
    for g in grid:
        rel = g - position
        shortest = min([abs(rel), abs(rel - z_length), abs(rel + z_length)])
        result.append(gaussian(shortest))

    return result


def rescale_distribution(dist: List[float], average: float) -> List[float]:
    scale = average / np.mean(dist)
    return (np.array(dist) * scale).tolist()

