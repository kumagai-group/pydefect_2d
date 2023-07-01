# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple

import numpy as np
from monty.json import MSONable
from numpy import linspace, pi, cos, sin


@dataclass
class Grid(MSONable):
    length: float  # in Å
    num_grid: int

    @cached_property
    def grid_points(self):
        return list(linspace(0, self.length, self.num_grid, endpoint=False))

    @property
    def mesh_dist(self):
        return self.length / self.num_grid


@dataclass
class Grids(MSONable):
    grids: List[Grid]
    ab_angle: float = 90

    def __call__(self, *args, **kwargs):
        return self.grids

    @property
    def x_grid(self):
        return self.grids[0]

    @property
    def y_grid(self):
        return self.grids[1]

    @property
    def grid_volume(self):
        return np.prod([grid.mesh_dist for grid in self.grids])

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
        return self.x_grid.length*self.y_grid.length * sin(self.ab_angle/180*pi)

    @property
    def gamma(self):
        return self.ab_angle / 180. * pi

    @property
    def cos_gamma(self):
        return cos(self.gamma)

    def squared_xy_grid_length(self, ix, iy):
        candidates = []
        theta = self.ab_angle/180*pi
        for rx, ry in itertools.product([0, -1], [0, -1]):
            x = self.x_grid.grid_points[ix] + self.x_grid.length * rx
            y = self.y_grid.grid_points[iy] + self.y_grid.length * ry
            candidates.append(x**2 + y**2 + 2*x*y*cos(self.gamma))

        xx = self.x_grid.grid_points[ix]
        yy = self.y_grid.grid_points[iy]

        return xx + yy*cos(theta), yy*sin(theta), min(candidates)

    @property
    def z_length(self):
        return self.grids[2].length

    @property
    def z_grid_points(self):
        return self.grids[2].grid_points

    @property
    def volume(self):
        return self.z_length * self.xy_area

    def nearest_z_grid_point(self, z) -> Tuple[int, float]:
        """
         :returns
            Tuple of nearest index and its z value.
        """
        return min(enumerate(self.z_grid_points), key=lambda x: abs(x[1]-z))