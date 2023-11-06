# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from math import pi, exp
from typing import Tuple

import numpy as np
from monty.json import MSONable
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.three_d.grids import Grid
from pydefect_2d.util.utils import with_end_point


@dataclass
class OneDGaussChargeModel(MSONable, ToJsonFileMixIn):
    """Gauss charge model with 1|e| under periodic boundary condition. """
    grid: Grid
    std_dev: float
    gauss_pos_in_frac: float  # in fractional coord. x=y=0
    surface: float  # in Å^2
    periodic_charges: np.array = None

    def __post_init__(self):
        if self.periodic_charges is None:
            self.periodic_charges = self._make_periodic_gauss_charges

    @property
    def _make_periodic_gauss_charges(self):
        coefficient = 1 / self.std_dev / (2 * pi) ** 0.5 / self.surface

        gauss = np.zeros(self.grid.num_grid)
        for nz, lz in enumerate(self.grid.grid_points()):
            gauss[nz] = exp(-self._min_z(lz) ** 2 / (2 * self.std_dev ** 2))

        return coefficient * gauss

    def _min_z(self, lz):
        return min([abs(lz - self.grid.length * (i + self.gauss_pos_in_frac))
                    for i in [-1, 0, 1]])

    @property
    def farthest_z_from_defect(self) -> Tuple[int, float]:
        rel_z_in_frac = (self.gauss_pos_in_frac + 0.5) % 1.
        z = self.grid.length * rel_z_in_frac
        return self.grid.nearest_grid_point(z)

    def to_plot(self, ax):
        ax.set_ylabel("Charge (|e|/Å)")
        xs = self.grid.grid_points(True)
        ys = with_end_point(self.periodic_charges * self.surface)
        ax.plot(xs, ys, label="charge", color="black")