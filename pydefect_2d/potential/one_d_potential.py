# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from abc import ABC
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
    """Abstract class for 1D potential profile"""
    grid: Grid
    potential: np.ndarray
    gauss_pos: float = None  # in fractional coordinate

    @property
    def grid_points(self):
        return self.grid.grid_points(end_point=True)

    @property
    def potential_w_end(self):
        return np.append(self.potential, self.potential[0])

    @cached_property
    def potential_func(self):
        return interp1d(self.grid_points, self.potential_w_end)

    def to_plot(self, ax):
        ax.set_ylabel("Potential (V)")
        ax.plot(self.grid_points, self.potential_w_end,
                label="potential", color="blue")


class Fp1DPotential(OneDPotential):
    pass


class Gauss1DPotential(OneDPotential):
    pass


@dataclass
class OneDPotDiff(MSONable, ToJsonFileMixIn):
    """Potential difference used for determining gaussian position"""
    fp_pot: Fp1DPotential
    gauss_pot: Gauss1DPotential

    # def __post_init__(self):
    #     assert self.fp_pot.grid == self.gauss_pot.grid

    @property
    def fp_grid_points(self):
        return self.fp_pot.grid.grid_points(end_point=False)

    @property
    def potential_diff_gradient(self):
        pos = self.gauss_pot.gauss_pos * self.fp_pot.grid.length
        idx, z = self.fp_pot.grid.farthest_grid_point(pos)

        z_m1 = self.fp_grid_points[idx - 1]
        z_p1 = self.fp_grid_points[idx + 1]
        diff1 = (self.gauss_pot.potential_func(z_m1)
                 - self.fp_pot.potential[idx-1])
        diff2 = (self.gauss_pot.potential_func(z_p1)
                 - self.fp_pot.potential[idx+1])
        return (diff2 - diff1) / (self.fp_pot.grid.mesh_dist * 2)


@dataclass
class PotDiffGradients(MSONable, ToJsonFileMixIn):
    gradients: List[float]
    gauss_positions: List[float]  # in fractional coordinate

    def to_plot(self, ax):
        ax.set_xlabel("Gradient (V/Å)")
        ax.set_ylabel("Gauss charge position (Å)")
        ax.plot(self.gauss_positions, self.gradients, color="blue")

    def gauss_pos_w_min_grad(self):
        idx = np.argmin(abs(np.array(self.gradients)))
        return self.gauss_positions[idx]
