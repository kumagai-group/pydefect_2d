# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import List

import numpy as np
from monty.json import MSONable
from scipy.interpolate import interp1d
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.potential.grids import Grid


@dataclass
class OneDPotential(MSONable, ToJsonFileMixIn, ABC):
    grid: Grid
    _potential: np.ndarray

    @abstractmethod
    def potential(self):
        pass

    @cached_property
    def interpol_pot_func(self):
        return interp1d(self.grid.grid_points, self.potential)

    def to_plot(self, ax):
        ax.set_ylabel("Potential (V)")
        ax.plot(self.grid.grid_points, self.potential,
                label="potential", color="blue")


@dataclass
class Fp1DPotential(OneDPotential):
    charge_state: int
    gauss_pos: float = None  # in frac

    @property
    def potential(self):
        return self._potential


@dataclass
class Gauss1DPotential(OneDPotential):
    gauss_pos: float  # in frac
    charge_state: int = None

    @property
    def potential(self):
        charge = self.charge_state or 1
        return self._potential * charge


@dataclass
class OneDPotDiff(MSONable, ToJsonFileMixIn):
    fp_pot: Fp1DPotential
    gauss_pot: Gauss1DPotential

    def __post_init__(self):
        assert self.fp_pot.grid == self.gauss_pot.grid
        assert self.fp_pot.charge_state == self.gauss_pot.charge_state

    @property
    def grid(self):
        return self.fp_pot.grid

    @property
    def pot_diff_grad(self):
        idx = self.grid.farthest_grid_point(self.gauss_pot.gauss_pos,
                                            in_frac_coords=True)
        val1 = self.gauss_pot.potential[idx-1] - self.fp_pot.potential[idx-1]
        val2 = self.gauss_pot.potential[idx+1] - self.fp_pot.potential[idx+1]
        return (val2 - val1) / (self.grid.mesh_dist * 2)


@dataclass
class PotDiffGradDist(MSONable, ToJsonFileMixIn):
    gradients: List[float]  # in frac coord
    gaussian_pos: List[float]  # in frac coord

    @cached_property
    def interpol_pot_func(self):
        return interp1d(self.gaussian_pos, self.gaussian_pos)

    def to_plot(self, ax):
        ax.set_xlabel("Gradient")
        ax.set_ylabel("Extrema position (Å)")
        ax.plot(self.gaussian_pos, self.gradients, color="blue")

    def gauss_pos_from_min_grad(self, num_grid=100):
        idx = np.argmin(abs(np.array(self.gradients)))
        return self.gaussian_pos[idx]
        # min_, max_ = np.min(self.gaussian_pos), np.max(self.gaussian_pos)
        # zs = np.linspace(min_, max_, num_grid, endpoint=False)
        # fs = self.interpol_pot_func(zs[1:])
        # idx = np.argmin(abs(fs))
        # return zs[idx]


# @dataclass
# class ExtremaDist(MSONable, ToJsonFileMixIn):
#     """Store extrema positions in vacuum as a function of gaussian site

    # Note:
    # In this class, periodicity is not checked. For example, if the extrema =
    # [-0.1, 0.0, 0.1], the class might not behave as expected.
    # """

    # extrema: List[float]  # in frac coord
    # gaussian_pos: List[float]  # in frac coord

    # @cached_property
    # def interpol_pot_func(self):
    #     return interp1d(self.gaussian_pos, self.gaussian_pos)

    # def to_plot(self, ax):
    #     ax.set_xlabel("Gauss charge position (Å)")
    #     ax.set_ylabel("Extrema position (Å)")
    #     ax.plot(self.gaussian_pos, self.extrema, color="blue")

    # def gauss_pos_from_extremum(self, extremum, num_grid=100):
    #     min_, max_ = np.min(self.gaussian_pos), np.max(self.gaussian_pos)
    #     zs = np.linspace(min_, max_, num_grid, endpoint=False)
    #     fs = self.interpol_pot_func(zs[1:]) - extremum
    #     idx = np.argmin(abs(fs))
    #     return zs[idx]


