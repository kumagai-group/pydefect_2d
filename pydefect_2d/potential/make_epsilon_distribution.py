# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from functools import cached_property
from typing import List

import numpy as np
from monty.json import MSONable
from numpy import linspace
from scipy.fftpack import fft
from tabulate import tabulate
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.potential.distribution import make_gaussian_distribution, \
    rescale_distribution


@dataclass
class Grid(MSONable):
    length: float  # in Å
    num_grid: int

    @property
    def grid_points(self):
        return list(linspace(0, self.length, self.num_grid, endpoint=False))


@dataclass
class EpsilonDistribution(MSONable, ToJsonFileMixIn):
    grid: Grid
    electronic: List[List[float]]  # [epsilon_x, epsilon_y, epsilon_z]
    ionic: List[List[float]]  # [epsilon_x, epsilon_y, epsilon_z]
    center: float  # in Å

    @property
    def ion_clamped(self) -> List[List[float]]:
        return list(np.array(self.electronic) + 1.)

    @property
    def static(self):
        return list(np.array(self.ion_clamped) + np.array(self.ionic))

    @property
    def effective(self):
        clamped = np.array(self.ion_clamped)
        ionic = np.array(self.ionic)
        return list(clamped + clamped**2/ionic)

    def __str__(self):
        result = [f"center: {self.center:.2f} Å"]
        header = ["pos (Å)"]
        for e in ["ε_inf", "ε_ion", "ε_0"]:
            for direction in ["x", "y", "z"]:
                header.append(f"{e}_{direction}")
        list_ = []
        for i, pos in enumerate(self.grid.grid_points):
            list_.append([pos])
            for e in [self.ion_clamped, self.ionic, self.static]:
#            for e in [self.ion_clamped, self.ionic, self.static, self.effective]:
                list_[-1].extend([e[0][i], e[1][i], e[2][i]])

        result.append(tabulate(list_, tablefmt="plain", floatfmt=".2f",
                               headers=header))

        return "\n".join(result)

    def to_plot(self, plt):
        ax = plt.gca()
        ax.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
        for e, direction in zip(self.ion_clamped, ["x", "y", "z"]):
            ax.plot(self.grid, e, label=f"ε_inf_{direction}")
        for e, direction in zip(self.ionic, ["x", "y", "z"]):
            ax.plot(self.grid, e, label=f"ε_ion_{direction}")
        ax.legend()

    @cached_property
    def reciprocal_static(self):
        return [fft(e) for e in self.static]


def make_gaussian_epsilon_distribution(length: float,
                                       num_gird: int,
                                       ave_electronic_epsilon: List[float],
                                       ave_ionic_epsilon: List[float],
                                       position: float,
                                       sigma: float):
    grid = Grid(length, num_gird)
    dist = make_gaussian_distribution(grid.grid_points, position, sigma)
    electronic = [rescale_distribution(dist, ave_ele)
                  for ave_ele in ave_electronic_epsilon]
    ionic = [rescale_distribution(dist, ave_ionic)
             for ave_ionic in ave_ionic_epsilon]
    return EpsilonDistribution(grid, electronic, ionic, position)


def make_large_model(epsilon_dist: EpsilonDistribution,
                     mul: int):



    return EpsilonDistribution(grid=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                               ion_clamped=[[1., 2., 2., 1., 1., 1.]] * 3,
                               ionic=[[0., 2., 2., 0., 0., 0.]] * 3,
                               center=epsilon_dist.center)


