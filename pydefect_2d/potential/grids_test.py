# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from math import sqrt, pi, sin

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from pydefect_2d.potential.grids import Grid, XYGrids, reduced_zone_idx


@pytest.fixture
def grid():
    return Grid(length=2.0, num_grid=4)


@pytest.fixture
def in_plane_grid(grid):
    return XYGrids(lattice=np.array([[10.0, 0.0], [-5.0, 5 * sqrt(3)]]),
                   num_grids=[4, 4])


def test_grid_properties(grid):
    assert_array_almost_equal(grid.grid_points, np.array([0.0, 0.5, 1.0, 1.5]))


def test_grid_neighboring_grid_idx(grid):
    assert grid.farthest_grid_point(0.3) == 3
    assert grid.farthest_grid_point(2.3) == 3
    assert grid.farthest_grid_point(0.3751, in_frac_coords=True) == 0
    assert grid.farthest_grid_point(0.3749, in_frac_coords=True) == 3
    assert grid.farthest_grid_point(0.8751, in_frac_coords=True) == 2


def test_in_plane_grids_rec_lattice(in_plane_grid):
    norm = 2*pi/10
    sin_theta = sin(120/180*pi)
    expected = [[norm*sqrt(3)/2/sin_theta, norm/2/sin_theta],
                [0., norm/sin_theta]]
    assert_almost_equal(in_plane_grid.rec_lattice, np.array(expected))


@pytest.fixture
def grids():
    return XYGrids(lattice=np.array([[6.0, 0.0], [-3.0, 3 * sqrt(3)]]),
                   num_grids=[3, 3])


def test_in_plane_grids_squared_length_on_girds(grids):
    actual = grids.grid_points
    s3, s3_2 = sqrt(3), sqrt(3) * 2
    expected = np.array([[[0.0, 0.0], [-1.0, s3], [-2.0, s3_2]],
                         [[2.0, 0.0], [1.0, s3], [0.0, s3_2]],
                         [[4.0, 0.0], [3.0, s3], [2.0, s3_2]]])
    assert_almost_equal(actual, expected)

    actual = grids.squared_length_on_grids
    expected = np.array([[0.0, 4.0, 4.0], [4.0, 4.0, 12.0], [4.0, 12.0, 4.0]])
    assert_almost_equal(actual, expected)


def test_reduced_zone_idx():
    assert_array_almost_equal(reduced_zone_idx(3), [0, 1, -1])
    assert_array_almost_equal(reduced_zone_idx(4), [0, 1, 2, -1])


def test_in_plane_grids_Ga2(grids: XYGrids):
    actual = grids.Ga2
    side_length = 6.0
    sin_theta = sin(120/180*pi)
    unit = (2 * pi / side_length / sin_theta) ** 2
    expected = np.array([0.0, unit, unit])
    assert_almost_equal(actual, expected)


