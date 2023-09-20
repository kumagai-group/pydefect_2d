# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from math import pi, exp

import numpy as np

from pydefect_2d.potential.grids import Grid


@dataclass
class OneDGaussChargeModel:
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


