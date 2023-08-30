# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import itertools
from dataclasses import dataclass
from functools import cached_property
from math import ceil
from typing import List, Tuple

import numpy as np
from monty.json import MSONable
from numpy import linspace, pi
from numpy.linalg import inv, norm


def reduced_zone_indices(num_grid) -> np.array:
    """
    :param num_grid:
    :return:
        reduced zone indices

    Example:
       num_grid = 4 -> [0, 1, 2, -1]
       num_grid = 5 -> [0, 1, 2, -2, -1]
    """
    result = np.array(range(num_grid), dtype=int)
    if num_grid % 2 == 0:
        mid_point = int(num_grid / 2 + 1)
    else:
        mid_point = int((num_grid + 1) / 2)
    result[mid_point:] -= num_grid
    return result


@dataclass
class Grid(MSONable):
    length: float  # in Å
    num_grid: int

    @classmethod
    def from_mesh_distance(cls, length, mesh_distance):
        num_grid = round(length / (mesh_distance * 2)) * 2
        return cls(length, num_grid)

    def __str__(self):
        return f"length: {self.length}, num grid: {self.num_grid}"

    def grid_points(self, end_point=False) -> np.ndarray:
        num_grid = self.num_grid + 1 if end_point else self.num_grid
        return linspace(0, self.length, num_grid, endpoint=end_point)

    @cached_property
    def Gs(self) -> np.ndarray:
        return 2 * pi / self.length * reduced_zone_indices(self.num_grid)

    @property
    def mesh_dist(self) -> float:
        return self.length / self.num_grid

    def nearest_grid_point(self, z) -> Tuple[int, float]:
        """
         :returns
            Tuple of nearest index and its z value.
        """
        idx = round(z / self.length * self.num_grid) % self.num_grid
        pt = self.mesh_dist * idx
        return idx, pt

    def farthest_grid_point(self, z) -> Tuple[int, float]:
        return self.nearest_grid_point(z - self.length / 2)


@dataclass
class XYGrids(MSONable):
    lattice: np.ndarray
    num_grids: List[int]

    @property
    def a_lat(self) -> np.ndarray:
        return self.lattice[0]

    @property
    def b_lat(self) -> np.ndarray:
        return self.lattice[1]

    @property
    def a_num_grid(self):
        return self.num_grids[0]

    @property
    def b_num_grid(self):
        return self.num_grids[1]

    @cached_property
    def rec_lattice(self) -> np.ndarray:
        return inv(self.lattice).T * 2 * pi

    @property
    def rec_a_lat(self) -> np.array:
        return self.rec_lattice[0]

    @property
    def rec_b_lat(self) -> np.array:
        return self.rec_lattice[1]

    @cached_property
    def Ga2(self) -> np.array:
        """1D np array of square of reciprocal vector along a-axis"""
        min_Ga_2 = np.inner(self.rec_a_lat, self.rec_a_lat)
        return min_Ga_2 * reduced_zone_indices(self.a_num_grid) ** 2

    @cached_property
    def Gb2(self) -> np.array:
        """1D np array of square of reciprocal vector along b-axis"""
        min_Gb_2 = np.inner(self.rec_b_lat, self.rec_b_lat)
        return min_Gb_2 * reduced_zone_indices(self.b_num_grid) ** 2

    @property
    def x_cross_y(self):
        return np.cross(self.a_lat, self.b_lat)

    @property
    def xy_area(self):
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

    def squared_length_on_grid(self, i, j, x=0.0, y=0.0):
        """
        :param i: The a index of the grid.
        :param j: The b index of the grid.
        :param x: X coordinate of  the origin in cartesian coord
        :param y: Y coordinate of  the origin in cartesian coord
        :return: Squared distance from the given origin to a specific grid point
        """
        results = []
        for rx, ry in itertools.product([1, 0, -1], [1, 0, -1]):
            rel_coords = (self.grid_points[i, j]
                          + self.a_lat * rx
                          + self.b_lat * ry
                          - np.array([x, y]))
            results.append(np.inner(rel_coords, rel_coords))
        return min(results)

    def squared_length_on_grids(self, x=0.0, y=0.0):
        """All the squared distances from the given origin to all grid points
        """
        result = np.zeros(self.num_grids)
        for xi in range(self.a_num_grid):
            for yi in range(self.b_num_grid):
                result[xi, yi] = self.squared_length_on_grid(xi, yi, x, y)
        return result

    def cart_to_frac(self, x: float, y: float) -> Tuple[float, float]:
        """Transform the given cartesian coordinates to fractional one."""
        [[X1, Y1], [X2, Y2]] = self.lattice
        a = (x * Y2 - y * X2) / (X1 * Y2 - X2 * Y1)
        b = (x * Y1 - y * X1) / (X2 * Y1 - X1 * Y2)
        return a, b

    def nearest_grid_point(
            self, x: float, y: float) -> Tuple[Tuple[int, int],
                                               Tuple[float, float],
                                               float]:
        """
         :returns
            ((indices), (cartesian coords), square length to the input point)
        """
        a, b = self.cart_to_frac(x, y)
        i = round(a * self.a_num_grid) % self.a_num_grid
        j = round(b * self.b_num_grid) % self.b_num_grid
        grid_point = self.grid_points[i, j]
        l2 = self.squared_length_on_grid(i, j, x, y)
        return ((i, j), tuple(grid_point), l2)


@dataclass
class Grids(MSONable):
    xy_grids: XYGrids
    z_grid: Grid

    @classmethod
    def from_z_grid(cls, xy_lat_matrix: np.ndarray, z_grid: Grid):
        a_lat, b_lat = norm(xy_lat_matrix[0]), norm(xy_lat_matrix[1])
        z_num_grid = z_grid.num_grid

        def ceil_to_even_number(lat):
            return ceil(lat / z_grid.length * z_num_grid / 2) * 2

        a_num_grid = ceil_to_even_number(a_lat)
        b_num_grid = ceil_to_even_number(b_lat)

        return Grids(xy_grids=XYGrids(lattice=xy_lat_matrix,
                                      num_grids=[a_num_grid, b_num_grid]),
                     z_grid=z_grid)

    @property
    def volume(self):
        return self.xy_grids.xy_area * self.z_grid.length
