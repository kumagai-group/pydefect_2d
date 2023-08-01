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
class OneDimPotential(MSONable, ToJsonFileMixIn):
    grid: Grid
    potential: np.ndarray
    charge_state: int = None

    @cached_property
    def interpol_pot_func(self):
        return interp1d(self.grid.grid_points_w_end, self.potential)

    def to_plot(self, ax):
        ax.set_ylabel("Potential (V)")
        ax.plot(self.grid.grid_points, self.potential,
                label="potential", color="blue")

    @property
    def vac_extremum_pot_pt(self):
        zs = np.linspace(0, self.grid.length, 1000, endpoint=False)
        charge = self.charge_state or 1
        fs = self.interpol_pot_func(zs) * charge
        idx = np.argmin(fs)
        return zs[idx]


@dataclass
class ExtremaDist(MSONable, ToJsonFileMixIn):
    """Positions of extrema points in vacuum as function of gaussian charge site
    """
    extrema: List[float]
    gaussian_pos: List[float]

    @cached_property
    def interpol_pot_func(self):
        return interp1d(self.gaussian_pos, self.gaussian_pos)

    def to_plot(self, ax):
        ax.set_xlabel("Gauss charge position (Å)")
        ax.set_ylabel("Extrema position (Å)")
        ax.plot(self.gaussian_pos, self.extrema, color="blue")

    def gauss_pos_from_extremum(self, extremum):
        min_, max_ = np.min(self.gaussian_pos), np.max(self.gaussian_pos)
        zs = np.linspace(min_, max_, 100, endpoint=False)
        fs = self.interpol_pot_func(zs[1:]) - extremum
        idx = np.argmin(abs(fs))
        return zs[idx]


