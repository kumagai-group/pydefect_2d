# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from abc import abstractmethod
from dataclasses import dataclass
from math import erf, exp

import numpy as np
from vise.util.logger import get_logger

from pydefect_2d.potential.grids import Grid


logger = get_logger(__name__)


@dataclass
class Dist(Grid):

    @property
    @abstractmethod
    def unscaled_dist(self) -> np.ndarray:
        pass

    def diele_in_plane_scale(self, ave_diele: float) -> np.ndarray:
        scale = (ave_diele - 1.) / self.unscaled_dist.mean()
        return self.unscaled_dist * scale + 1.

    def diele_out_of_plane_scale(self,
                                 ave_diele: float,
                                 reduction_ratio: float = 0.9,
                                 convergence_ratio: float = 10 ** -6,
                                 max_iteration: int = 100) -> np.ndarray:
        """Calculate the scaled distribution

        static = 1 / ((1 + factor * unscaled_dist)).mean()

        :param ave_diele:  # w/o vacuum permittivity
        :param reduction_ratio:
        :param convergence_ratio:
        :param max_iteration:

        :return:
        """
        scale_factor = (ave_diele - 1.) / self.unscaled_dist.mean()
        del_scale_factor = scale_factor * reduction_ratio

        for i in range(max_iteration):
            aaa = 1.0 / (1. + scale_factor * self.unscaled_dist)
            unscale_mean = 1 / aaa.mean()
            ratio = (ave_diele - unscale_mean) / ave_diele
            if abs(ratio) < convergence_ratio:
                break
            # need to increase unscale_mean = reduce alpha
            ratio_sign = int(ratio > 0) - int(ratio < 0)
            scale_factor += del_scale_factor * ratio_sign
            del_scale_factor *= reduction_ratio
        else:
            logger.warning("No convergence is reached. The number of iteration "
                           f"is {max_iteration}, and the convergence ratio is "
                           f"{abs(ratio)} > {convergence_ratio}")
            raise ValueError("No convergence is reached.")

        return scale_factor * self.unscaled_dist + 1.0


@dataclass
class ManualDist(Dist):
    manual_dist: np.ndarray

    @property
    def unscaled_dist(self) -> np.ndarray:
        return self.manual_dist


@dataclass
class GaussianDist(Dist):
    center: float  # in Å
    sigma: float  # in Å

    @classmethod
    def from_grid(cls, grid: Grid, center, sigma):
        return cls(grid.length, grid.num_grid, center, sigma)

    def __str__(self):
        result = [super().__str__(),
                  f"center: {self.center:.1} Å",
                  f"sigma: {self.sigma:.1} Å"]
        return "\n".join(result)

    @property
    def unscaled_dist(self) -> np.ndarray:
        """Distribution w/o normalization under periodic boundary condition.

        All lengths are in Å.
        """
        def gaussian(length):
            return exp(-length**2/(2*self.sigma**2))

        result = []
        for g in self.grid_points:
            rel = g - self.center
            shortest = min([abs(rel),
                            abs(rel - self.length),
                            abs(rel + self.length)])
            result.append(gaussian(shortest))

        return np.array(result)


@dataclass
class StepDist(Dist):
    step_left: float  # in Å
    step_right: float  # in Å
    error_func_width: float  # in Å

    @classmethod
    def from_grid(cls, grid: Grid, step_left, step_right, error_func_width):
        return cls(grid.length, grid.num_grid, step_left, step_right,
                   error_func_width)

    def __str__(self):
        result = [f"step left: {self.step_left:.2f} Å",
                  f"step right: {self.step_right:.2f} Å",
                  f"width of error function: {self.error_func_width:.2f} Å",
                  super().__str__()]
        return "\n".join(result)

    @property
    def unscaled_dist(self) -> np.ndarray:
        """ Make step-like distribution

        :param grid: Cartesian coordinates in Å.
        :param step_left: Cartesian coord in Å
        :param step_right: Cartesian coord in Å
        :param error_func_width: Width in Å.

        """

        def func_left(dist):
            return - erf(dist / self.error_func_width) / 2 + 0.5

        def func_right(dist):
            return erf(dist / self.error_func_width) / 2 + 0.5

        result = []
        for g in self.grid_points:
            d = {"l": self.step_left - g,
                 "l_p1": self.step_left - g + self.length,
                 "l_m1": self.step_left - g - self.length,
                 "r": self.step_right - g,
                 "r_p1": self.step_right - g + self.length,
                 "r_m1": self.step_right - g - self.length}
            dd = {k: abs(v) for k, v in d.items()}
            shortest = min(d, key=dd.get)

            if shortest[0] == "l":
                result.append(func_left(d[shortest]))
            else:
                result.append(func_right(d[shortest]))

        return np.array(result)


