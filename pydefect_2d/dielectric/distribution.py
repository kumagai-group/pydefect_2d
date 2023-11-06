# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from abc import abstractmethod
from dataclasses import dataclass
from math import erf, exp, sqrt
from typing import List

import numpy as np
from scipy.optimize import minimize
from vise.util.logger import get_logger

from pydefect_2d.three_d.grids import Grid


logger = get_logger(__name__)


@dataclass
class Dist(Grid):

    @property
    @abstractmethod
    def unscaled_in_plane_dist(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def unscaled_out_of_plane_dist(self) -> np.ndarray:
        pass

    @property
    def to_manual_dist(self) -> "ManualDist":
        return ManualDist(self.length,
                          self.num_grid,
                          self.unscaled_in_plane_dist,
                          self.unscaled_out_of_plane_dist)

    def diele_in_plane_scale(self, ave_diele: float) -> np.ndarray:
        scale = (ave_diele - 1.) / self.unscaled_in_plane_dist.mean()
        return self.unscaled_in_plane_dist * scale + 1.

    def diele_out_of_plane_scale(self, ave_diele: float) -> np.ndarray:
        """Calculate the scaled distribution

        static = 1 / ((1 + factor * unscaled_dist)).mean()

        :param ave_diele:  # w/o vacuum permittivity

        :return:
        """

        def f(factor):
            denominator = 1.0 / (1. + factor * self.unscaled_out_of_plane_dist)
            return abs(1 / denominator.mean() - ave_diele)

        initial_guess = \
            (ave_diele - 1.) / self.unscaled_out_of_plane_dist.mean()
        min_res = minimize(f, initial_guess, method='BFGS')
        scale_factor = min_res.x[0]
        return scale_factor * self.unscaled_out_of_plane_dist + 1.0

    @property
    def grid(self):
        return Grid(self.length, self.num_grid)


@dataclass
class ManualDist(Dist):
    unscaled_in_plane_dist_: np.ndarray
    unscaled_out_of_plane_dist_: np.ndarray

    def __add__(self, other: Dist):
        in_ = self.unscaled_in_plane_dist + other.unscaled_in_plane_dist
        out = self.unscaled_out_of_plane_dist + other.unscaled_out_of_plane_dist
        return ManualDist(self.length, self.num_grid, in_, out)

    def __post_init__(self):
        assert self.num_grid == len(self.unscaled_in_plane_dist_)
        assert self.num_grid == len(self.unscaled_out_of_plane_dist_)

    @property
    def unscaled_in_plane_dist(self) -> np.ndarray:
        return self.unscaled_in_plane_dist_

    @property
    def unscaled_out_of_plane_dist(self) -> np.ndarray:
        return self.unscaled_out_of_plane_dist_

    @classmethod
    def from_grid(cls, grid: Grid, in_plane_dist, out_of_plane_dist=None):
        if out_of_plane_dist is None:
            out_of_plane_dist = in_plane_dist
        return cls(grid.length, grid.num_grid, in_plane_dist, out_of_plane_dist)


@dataclass
class GaussianDist(Dist):
    center: float  # in Å
    in_plane_sigma: float  # in Å
    out_of_plane_sigma: float  # in Å

    @classmethod
    def from_grid(cls, grid: Grid, center, sigma):
        return cls(grid.length, grid.num_grid, center, sigma, sigma)

    def __str__(self):
        result = [super().__str__(),
                  f"center: {self.center:.1} Å",
                  f"in-plane sigma: {self.in_plane_sigma:.1} Å",
                  f"out-of-plane sigma: {self.out_of_plane_sigma:.1} Å"]
        return "\n".join(result)

    @property
    def unscaled_in_plane_dist(self) -> np.ndarray:
        return self.unscaled_dist(self.in_plane_sigma)

    @property
    def unscaled_out_of_plane_dist(self) -> np.ndarray:
        return self.unscaled_dist(self.out_of_plane_sigma)

    def unscaled_dist(self, sigma) -> np.ndarray:
        """Distribution w/o normalization under periodic boundary condition.

        All lengths are in Å.
        """
        def gaussian(length):
            return exp(-length**2/(2*sigma**2))

        result = []
        for g in self.grid_points():
            rel = g - self.center
            shortest = min([abs(rel),
                            abs(rel - self.length),
                            abs(rel + self.length)])
            result.append(gaussian(shortest))

        return np.array(result)


@dataclass
class PeriodicGaussianDist(Dist):
    centers: List[float]  # in Å
    sigma: float  # in Å

    @classmethod
    def from_grid(cls, grid: Grid, centers, sigma):
        return cls(grid.length, grid.num_grid, centers, sigma)

    @property
    def unscaled_in_plane_dist(self) -> np.ndarray:
        return self.unscaled_dist

    @property
    def unscaled_out_of_plane_dist(self) -> np.ndarray:
        return self.unscaled_dist

    @property
    def unscaled_dist(self) -> np.ndarray:
        """Distribution w/o normalization under periodic boundary condition.

        All lengths are in Å.
        """
        def gaussian(length):
            return exp(-length**2/(2*self.sigma**2))

        result = np.zeros(self.grid.num_grid)
        for g in self.grid_points():
            rel = g - self.center
            shortest = min([abs(rel),
                            abs(rel - self.length),
                            abs(rel + self.length)])
            result += gaussian(shortest)

        return np.array(result)


@dataclass
class StepDist(Dist):
    """ Make step-like distribution

    sigma: Width in Å.
    """
    center: float  # in Å
    in_plane_width: float  # in Å
    in_plane_sigma: float  # in Å
    out_of_plane_width: float  # in Å
    out_of_plane_sigma: float  # in Å

    @classmethod
    def from_grid(cls, grid: Grid, center: float, in_plane_width: float,
                  out_of_plane_width: float, sigma: float):
        return cls(
            grid.length, grid.num_grid, center,
            in_plane_width=in_plane_width, in_plane_sigma=sigma,
            out_of_plane_width=out_of_plane_width, out_of_plane_sigma=sigma)

    def __str__(self):
        result = [
            f""
            f"Center: {self.center:.2f} Å",
            f"In-plane width: {self.in_plane_width:.2f} Å",
            f"In-plane sigma of error func: {self.in_plane_sigma:.2f} Å",
            f"Out-of-plane width: {self.in_plane_width:.2f} Å",
            f"Out-of-plane sigma of error func: {self.in_plane_sigma:.2f} Å",
            super().__str__()]
        return "\n".join(result)

    @property
    def unscaled_in_plane_dist(self) -> np.ndarray:
        return self.unscaled_dist(self.in_plane_width, self.in_plane_sigma)

    @property
    def unscaled_out_of_plane_dist(self) -> np.ndarray:
        return self.unscaled_dist(
            self.out_of_plane_width, self.out_of_plane_sigma)

    def unscaled_dist(self, width, sigma) -> np.ndarray:
        error_func_width = sigma * sqrt(2)
        step_left = self.center - width / 2
        step_right = self.center + width / 2

        def func_left(dist):
            return - erf(dist / error_func_width) / 2 + 0.5

        def func_right(dist):
            return erf(dist / error_func_width) / 2 + 0.5

        result = []
        for g in self.grid_points():
            d = {"l": step_left - g,
                 "l_p1": step_left - g + self.length,
                 "l_m1": step_left - g - self.length,
                 "r": step_right - g,
                 "r_p1": step_right - g + self.length,
                 "r_m1": step_right - g - self.length}
            dd = {k: abs(v) for k, v in d.items()}
            shortest = min(d, key=dd.get)

            if shortest[0] == "l":
                result.append(func_left(d[shortest]))
            else:
                result.append(func_right(d[shortest]))

        return np.array(result)


