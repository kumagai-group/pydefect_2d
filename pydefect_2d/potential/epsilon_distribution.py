# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import List

import numpy as np
from monty.json import MSONable
from scipy.fftpack import fft
from tabulate import tabulate
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.potential.distribution import make_gaussian_distribution, \
    rescale_distribution, make_step_like_distribution
from pydefect_2d.potential.grids import Grid


@dataclass
class EpsilonDistribution(MSONable, ToJsonFileMixIn):
    """

    The z-direction is normal to the surfaces.
    """

    grid: Grid
    ave_electronic_epsilon: List[float]  # not include vacuum permittivity.
    ave_ionic_epsilon: List[float]

    @property
    @abstractmethod
    def unscaled_dist(self) -> np.ndarray:
        pass

    @property
    def ave_ele_ion(self) -> List[float]:
        return [e + i for e, i in zip(self.ave_electronic_epsilon,
                                      self.ave_ionic_epsilon)]

    @cached_property
    def static(self):
        return [
            rescale_distribution(self.unscaled_dist, self.ave_ele_ion[0]) + 1.,
            rescale_distribution(self.unscaled_dist, self.ave_ele_ion[1]) + 1.,
            scaling_z_direction(self.unscaled_dist, self.ave_ele_ion[2]) + 1.
        ]

    @cached_property
    def reciprocal_static(self):
        return [fft(e) for e in self.static]

    def __str__(self):
        result = []
        header = ["pos (Å)"]
        for e in ["ε_0"]:
            for direction in ["x", "y", "z"]:
                header.append(f"{e}_{direction}")
        list_ = []
        for i, pos in enumerate(self.grid.grid_points):
            list_.append([pos])
            for e in [self.static]:
                list_[-1].extend([e[0][i], e[1][i], e[2][i]])

        result.append(tabulate(list_, tablefmt="plain", floatfmt=".2f",
                               headers=header))

        return "\n".join(result)

    def to_plot(self, ax):
        ax.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
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


def scaling_z_direction(unscaled_dist: np.array,
                        ave_diele: np.array,
                        reduction_ratio: float = 0.9,
                        convergence_ratio: float = 10**-6,
                        max_iteration: int = 100) -> np.array:
    """Calculate the scaled distribution

   \epsilon^-1 = ((1 + factor * unscaled_dist)).mean()

    :param unscaled_dist:   # w/o vacuum permittivity
    :param ave_diele:  # w/o vacuum permittivity
    :param reduction_ratio:
    :param convergence_ratio:
    :param max_iteration:

    :return:
    """
    factor = d_factor = ave_diele / unscaled_dist.mean()
    inv_diele = 1 / (ave_diele + 1.)

    for i in range(max_iteration):
        unscale_mean = (1. / (1. + factor * unscaled_dist)).mean()
        f = (inv_diele - unscale_mean) / inv_diele
        if abs(f) < convergence_ratio:
            break
        # need to increase unscale_mean = reduce alpha
        f_sign = int(f > 0) - int(f < 0)
        factor -= d_factor * f_sign
        d_factor *= reduction_ratio
    else:
        raise ValueError("No convergence is reached.")

    return factor * unscaled_dist  # w/o vacuum
