# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from functools import cached_property
from typing import List

import numpy as np
from monty.json import MSONable
from scipy.interpolate import interp1d
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.potential.grids import Grid


@dataclass
class OneDPotential(MSONable, ToJsonFileMixIn):
    grid: Grid
    potential: np.ndarray
    charge_state: int = None
    gauss_pos: float = None

    @cached_property
    def interpol_pot_func(self):
        return interp1d(self.grid.grid_points_w_end, self.potential)

    def to_plot(self, ax):
        ax.set_ylabel("Potential (V)")
        ax.plot(self.grid.grid_points, self.potential,
                label="potential", color="blue")

    @property
    def vac_extremum_pot_pt(self):
        """Potential extremum point in vacuum in frac coord."""
        zs = np.linspace(0, self.grid.length, 1000, endpoint=False)
        charge = self.charge_state or 1
        fs = self.interpol_pot_func(zs) * charge
        idx = np.argmin(fs)
        return zs[idx] / self.grid.length


@dataclass
class OneDPotDiff(MSONable, ToJsonFileMixIn):
    grid: Grid
    pot_1: np.ndarray
    pot_2: np.ndarray
    gauss_pos: float

    @property
    def vac_pot_gradient(self):

        return


@dataclass
class ExtremaDist(MSONable, ToJsonFileMixIn):
    """Store extrema positions in vacuum as a function of gaussian site

    Note:
    In this class, periodicity is not checked. For example, if the extrema =
    [-0.1, 0.0, 0.1], the class might not behave as expected.
    """

    extrema: List[float]  # in frac coord
    gaussian_pos: List[float]  # in frac coord

    @cached_property
    def interpol_pot_func(self):
        return interp1d(self.gaussian_pos, self.gaussian_pos)

    def to_plot(self, ax):
        ax.set_xlabel("Gauss charge position (Å)")
        ax.set_ylabel("Extrema position (Å)")
        ax.plot(self.gaussian_pos, self.extrema, color="blue")

    def gauss_pos_from_extremum(self, extremum, num_grid=100):
        min_, max_ = np.min(self.gaussian_pos), np.max(self.gaussian_pos)
        zs = np.linspace(min_, max_, num_grid, endpoint=False)
        fs = self.interpol_pot_func(zs[1:]) - extremum
        idx = np.argmin(abs(fs))
        return zs[idx]


