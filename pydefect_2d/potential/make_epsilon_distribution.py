# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from typing import List

import numpy as np
from monty.json import MSONable
from tabulate import tabulate
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.potential.distribution import make_gaussian_distribution, \
    rescale_distribution


@dataclass
class EpsilonDistribution(MSONable, ToJsonFileMixIn):
    grid: List[float]  # assume orthogonal system
    ion_clamped: List[List[float]]  # [epsilon_x, epsilon_y, epsilon_z]
    ionic: List[List[float]]  # [epsilon_x, epsilon_y, epsilon_z]

    @property
    def static(self):
        return list(np.array(self.ion_clamped) + np.array(self.ionic))

    @property
    def effective(self):
        clamped = np.array(self.ion_clamped)
        ionic = np.array(self.ionic)
        return list(clamped + clamped**2/ionic)

    def __str__(self):
        header = ["pos (Å)"]
        for e in ["ε_inf", "ε_ion", "ε_0"]:
            for direction in ["x", "y", "z"]:
                header.append(f"{e}_{direction}")
        list_ = []
        for i, pos in enumerate(self.grid):
            list_.append([pos])
            for e in [self.ion_clamped, self.ionic, self.static]:
#            for e in [self.ion_clamped, self.ionic, self.static, self.effective]:
                list_[-1].extend([e[0][i], e[1][i], e[2][i]])

        return tabulate(list_, tablefmt="plain", floatfmt=".2f", headers=header)

    def to_plot(self, ax):
        ax.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
        for e, direction in zip(self.ion_clamped, ["x", "y", "z"]):
            ax.plot(self.grid, e, label=f"ε_inf_{direction}")
        for e, direction in zip(self.ionic, ["x", "y", "z"]):
            ax.plot(self.grid, e, label=f"ε_ion_{direction}")
        ax.legend()


def make_gaussian_epsilon_distribution(grid: List[float],
                                       ave_ion_clamped_epsilon: List[float],
                                       ave_ionic_epsilon: List[float],
                                       position: float,
                                       sigma: float):
    dist = make_gaussian_distribution(grid, position, sigma)
    clamped = [rescale_distribution(dist, ave_clamped, False)
               for ave_clamped in ave_ion_clamped_epsilon]
    ionic = [rescale_distribution(dist, ave_ionic, True)
               for ave_ionic in ave_ionic_epsilon]
    return EpsilonDistribution(grid, clamped, ionic)
