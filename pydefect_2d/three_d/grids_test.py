# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from math import sqrt, pi, sin

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from pydefect_2d.three_d.grids import Grid, XYGrids, reduced_zone_indices, Grids


def test_reduced_zone_idx():
    assert_array_almost_equal(reduced_zone_indices(3), [0, 1, -1])
    assert_array_almost_equal(reduced_zone_indices(4), [0, 1, 2, -1])


@pytest.fixture
def grid():
    return Grid(length=2.0, num_grid=4)


def test_grid_points(grid):
    assert_array_almost_equal(grid.grid_points(),
                              np.array([0.0, 0.5, 1.0, 1.5]))
    assert_array_almost_equal(grid.grid_points(end_point=True),
                              np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert_array_almost_equal(grid.Gs, 2*pi/2*np.array([0, 1, 2, -1]))
    assert grid.mesh_dist == 0.5


def test_grid_nearest_point(grid):
    assert grid.nearest_grid_point(1.59) == (3, 1.5)


def test_grid_neighboring_grid_idx(grid):
    assert grid.farthest_grid_point(0.3) == (3, 1.5)
    assert grid.farthest_grid_point(0.8) == (0, 0.0)
    assert grid.farthest_grid_point(2.3) == (3, 1.5)


def test_grid_from_mesh_distance(grid):
    actual = Grid.from_mesh_distance(length=2.0, mesh_distance=0.5)
    assert actual == grid
    actual = Grid.from_mesh_distance(length=2.0, mesh_distance=0.65)
    assert actual == grid


@pytest.fixture
def xy_grids():
    return XYGrids(lattice=np.array([[6.0, 0.0], [-3.0, 3 * sqrt(3)]]),
                   num_grids=[3, 3])


def test_xy_grids_rec_lattice(xy_grids):
    norm = 2*pi/6
    sin_theta = sin(120/180*pi)
    expected = [[norm*sqrt(3)/2/sin_theta, norm/2/sin_theta],
                [0., norm/sin_theta]]
    assert_almost_equal(xy_grids.rec_lattice, np.array(expected))


def test_xy_grids_Ga2(xy_grids: XYGrids):
    side_length = 6.0
    sin_theta = sin(120/180*pi)
    unit = (2 * pi / side_length / sin_theta) ** 2
    expected = np.array([0.0, unit, unit])

    assert_almost_equal(xy_grids.Ga2, expected)
    assert_almost_equal(xy_grids.Gb2, expected)


def test_xy_grids_squared_length_on_girds(xy_grids):
    actual = xy_grids.grid_points
    s3, s3_2 = sqrt(3), sqrt(3) * 2
    expected = np.array([[[0.0, 0.0], [-1.0, s3], [-2.0, s3_2]],
                         [[2.0, 0.0], [1.0, s3], [0.0, s3_2]],
                         [[4.0, 0.0], [3.0, s3], [2.0, s3_2]]])
    assert_almost_equal(actual, expected)

    actual = xy_grids.squared_length_on_grids()
    expected = np.array([[0.0, 4.0, 4.0], [4.0, 4.0, 12.0], [4.0, 12.0, 4.0]])
    assert_almost_equal(actual, expected)


def test_xy_grids_nearest():
    xy_grids = XYGrids(lattice=np.array([[4.0, 0.0], [-2.0, 2 * sqrt(3)]]),
                       num_grids=[4, 4])
    actual = xy_grids.nearest_grid_point(1.0, sqrt(3) * 2 - 0.1)
    expected = ((3, 0), (3.0, 0.0), 0.010000000000000018)
    assert actual == expected


def test_grids_from_z_num_grid():
    xy_lat_matrix = np.array([[6.0, 0.0], [-3.0, 3 * sqrt(3)]])
    z_grid = Grid(length=10.0, num_grid=5)

    actual = Grids.from_z_grid(xy_lat_matrix, z_grid)

    xy_grids = XYGrids(lattice=xy_lat_matrix, num_grids=[4, 4])
    expected = Grids(xy_grids=xy_grids, z_grid=z_grid)

    assert actual == expected




