# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import List

import numpy as np
from monty.json import MSONable
from numpy.testing import assert_almost_equal
from scipy.fftpack import fft
from tabulate import tabulate
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.potential.distribution import make_gaussian_distribution, \
    rescale_distribution, make_step_like_distribution
from pydefect_2d.potential.grids import Grid


@dataclass
class EpsilonDistribution(MSONable, ToJsonFileMixIn):
    grid: Grid
    ave_electronic_epsilon: List[float]
    ave_ionic_epsilon: List[float]

    @property
    @abstractmethod
    def unscaled_dist(self) -> np.ndarray:
        pass

    def __eq__(self, other):
        try:
            assert self.grid == other.grid
            assert_almost_equal(self.electronic, other.electronic)
            assert_almost_equal(self.ionic, other.ionic)
        except AssertionError:
            return False
        return True

    @property
    def electronic(self):
        return np.array([rescale_distribution(self.unscaled_dist, ave_ele)
                         for ave_ele in self.ave_electronic_epsilon])

    @property
    def ionic(self):
        return np.array([rescale_distribution(self.unscaled_dist, ave_ionic)
                         for ave_ionic in self.ave_ionic_epsilon])

    @cached_property
    def ion_clamped(self):
        return self.electronic + 1.

    @cached_property
    def static(self):
        return self.ion_clamped + self.ionic

    @cached_property
    def effective(self):
        return self.ion_clamped + self.ion_clamped ** 2 / self.ionic

    @cached_property
    def reciprocal_static(self):
        return [fft(e) for e in self.static]

    def __str__(self):
        result = []
        header = ["pos (Å)"]
        for e in ["ε_inf", "ε_ion", "ε_0"]:
            for direction in ["x", "y", "z"]:
                header.append(f"{e}_{direction}")
        list_ = []
        for i, pos in enumerate(self.grid.grid_points):
            list_.append([pos])
            for e in [self.ion_clamped, self.ionic, self.static]:
                list_[-1].extend([e[0][i], e[1][i], e[2][i]])

        result.append(tabulate(list_, tablefmt="plain", floatfmt=".2f",
                               headers=header))

        return "\n".join(result)

    def to_plot(self, plt):
        ax = plt.gca()
        ax.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
        for e, direction in zip(self.ion_clamped, ["x", "y", "z"]):
            ax.plot(self.grid.grid_points, e, label=f"ε_inf_{direction}")
        for e, direction in zip(self.ionic, ["x", "y", "z"]):
            ax.plot(self.grid.grid_points, e, label=f"ε_ion_{direction}")
        for e, direction in zip(self.static, ["x", "y", "z"]):
            ax.plot(self.grid.grid_points, e, label=f"ε_0_{direction}")
        ax.legend()


@dataclass
class EpsilonGaussianDistribution(EpsilonDistribution):
    center: float  # in Å
    sigma: float  # in Å

    def __str__(self):
        result = [f"center: {self.center:.2f} Å",
                  f"sigma: {self.sigma:.2f} Å",
                  super().__str__()]
        return "\n".join(result)

    @property
    def unscaled_dist(self):
        return make_gaussian_distribution(self.grid, self.center, self.sigma)


@dataclass
class EpsilonStepLikeDistribution(EpsilonDistribution):
    step_left: float  # in Å
    step_right: float  # in Å
    error_func_width: float  # in Å

    def __str__(self):
        result = [f"step left: {self.step_left:.2f} Å",
                  f"step right: {self.step_right:.2f} Å",
                  f"width of error function: {self.error_func_width:.2f} Å",
                  super().__str__()]
        return "\n".join(result)

    @property
    def unscaled_dist(self):
        return make_step_like_distribution(self.grid, self.step_left,
                                           self.step_right,
                                           self.error_func_width)
