# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from math import erf
from typing import List

import numpy as np


def make_epsilon_distribution(dielectric_constants: List[float],
                              grid: np.ndarray,
                              z_latt: float,
                              slab_left: float,
                              slab_right: float,
                              error_func_width):
    """ Make dielectric constant (epsilon) distribution in 3D.

    :param dielectric_constants: Three values along x, y ,and z-directions.
    :param grid: Three values
    :param z_latt:
    :param slab_left:
    :param slab_right:
    :param error_func_width: Width in Å.

    :return:
    """

    def func_left(z):
        # e2 | 1.0
        return [(1.0 - e2) * erf(z / error_func_width) / 2 + (1.0 + e2) / 2
                for e2 in dielectric_constants]

    def func_right(z):
        # 1.0 | e2
        return [(e2 - 1.0) * erf(z / error_func_width) / 2 + (1.0 + e2) / 2
                for e2 in dielectric_constants]

    result_t = []
    for g in grid:
        d = {"l": slab_left - g,
             "l_p1": slab_left - g + z_latt,
             "l_m1": slab_left - g - z_latt,
             "r": slab_right - g,
             "r_p1": slab_right - g + z_latt,
             "r_m1": slab_right - g - z_latt}
        dd = {k: abs(v) for k, v in d.items()}
        shortest = min(d, key=dd.get)

        if shortest[0] == "l":
            result_t.append(func_left(d[shortest]))
        else:
            result_t.append(func_right(d[shortest]))

    return np.array(result_t).T.tolist()
