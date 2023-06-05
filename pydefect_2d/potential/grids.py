# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from monty.json import MSONable
from numpy import linspace


@dataclass
class Grid(MSONable):
    base_length: float  # in Å
    base_num_grid: int
    mul: int = 1

    @property
    def length(self):
        return self.base_length * self.mul

    @property
    def num_grid(self):
        return self.base_num_grid * self.mul

    @property
    def grid_points(self):
        return list(linspace(0, self.length, self.num_grid, endpoint=False))


@dataclass
class Grids(MSONable):
    grids: List[Grid]

    def __call__(self, *args, **kwargs):
        return self.grids

    @property
    def all_grid_points(self):
        return [g.grid_points for g in self.grids]

    @property
    def num_grid_points(self):
        return [grid.num_grid for grid in self.grids]

    @property
    def lengths(self):
        return [grid.length for grid in self.grids]

    @property
    def xy_area(self):
        return np.prod(self.lengths[:2])

    @property
    def z_length(self):
        return self.grids[2].length

    @property
    def z_grid_points(self):
        return self.grids[2].grid_points

    @property
    def volume(self):
        return np.prod(self.lengths)

    def nearest_z_grid_point(self, z) -> Tuple[int, float]:
        """
         :returns
            Tuple of nearest index and its z value.
        """
        return min(enumerate(self.z_grid_points), key=lambda x: abs(x[1]-z))