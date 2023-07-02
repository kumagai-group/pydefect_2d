# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import itertools
from dataclasses import dataclass
from functools import cached_property
from math import sqrt
from typing import List, Tuple

import numpy as np
from monty.json import MSONable
from numpy import linspace, pi, cos, sin, inner
from numpy.linalg import inv


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

    @cached_property
    def Gs(self):
        return 2 * pi / self.length * np.array(reduced_zone_idx(self.num_grid))


@dataclass
class XYGrids(MSONable):
    lattice: np.array
    num_grids: List[int]

    @property
    def a_lat(self):
        return self.lattice[0]

    @property
    def b_lat(self):
        return self.lattice[1]

    @property
    def a_num_grid(self):
        return self.num_grids[0]

    @property
    def b_num_grid(self):
        return self.num_grids[1]

    @property
    def x_cross_y(self):
        return np.cross(self.a_lat, self.b_lat)

    @property
    def area(self):
        return np.linalg.norm(self.x_cross_y)

    @cached_property
    def grid_points(self):
        # [[x_1, y_1], [x_2, y_2], ...]
        result = np.zeros(self.num_grids + [2])
        for xi in range(self.a_num_grid):
            for yi in range(self.b_num_grid):
                a = self.a_lat * xi / self.a_num_grid
                b = self.b_lat * yi / self.b_num_grid
                result[xi, yi] = a + b
        return np.array(result)

    @cached_property
    def squared_length_on_grids(self):

        result = np.zeros(self.num_grids)

        for xi in range(self.a_num_grid):
            for yi in range(self.b_num_grid):
                candidates = []
                for rx, ry in itertools.product([0, -1], [0, -1]):
                    coord = (self.grid_points[xi, yi]
                             + self.a_lat * rx + self.b_lat * ry)
                    candidates.append(inner(coord, coord))
                result[xi, yi] = min(candidates)

        return result

    @cached_property
    def rec_lattice(self):
        return inv(self.lattice).T * 2 * pi

    @property
    def rec_a_lat(self):
        return self.rec_lattice[0]

    @property
    def rec_b_lat(self):
        return self.rec_lattice[1]

    @cached_property
    def Ga2(self):
        Ga_2 = np.inner(self.rec_a_lat, self.rec_a_lat)
        return Ga_2 * np.array(reduced_zone_idx(self.a_num_grid)) ** 2

    @cached_property
    def Gb2(self):
        Gb_2 = np.inner(self.rec_b_lat, self.rec_b_lat)
        return Gb_2 * np.array(reduced_zone_idx(self.b_num_grid)) ** 2


def reduced_zone_idx(n_mesh):
    middle = int(n_mesh / 2)
    if n_mesh % 2 == 1:
        return list(range(middle + 1)) + list(range(middle, 0, -1))
    else:
        return list(range(middle + 1)) + list(range(middle - 1, 0, -1))


@dataclass
class Grids(MSONable):
    xy_grids: XYGrids
    z_grid: Grid

    @property
    def volume(self):
        return self.xy_grids.area * self.z_grid.length

    @property
    def z_length(self):
        return self.z_grid.length

    @property
    def z_grid_points(self):
        return self.z_grid.grid_points

    def nearest_z_grid_point(self, z) -> Tuple[int, float]:
        """
         :returns
            Tuple of nearest index and its z value.
        """
        return min(enumerate(self.z_grid_points), key=lambda x: abs(x[1]-z))