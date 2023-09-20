# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from pydefect_2d.dielectric.distribution import ManualDist
from pydefect_2d.dielectric.dielectric_distribution import \
    DielectricConstDist
from pydefect_2d.three_d.grids import Grid


grid = Grid(20., 2)


@pytest.fixture
def diele_dist():
    dist = ManualDist.from_grid(grid, np.array([0.0, 1.0]))
    return DielectricConstDist(
        ave_ele=[3., 3., 0.5], ave_ion=[1., 1., 1.0], dist=dist)


def test_diele_dist_properties(diele_dist: DielectricConstDist):
    assert_array_almost_equal(diele_dist.grid_points, [0.0, 10.0])
    assert diele_dist.ave_static_x == 4.
    assert diele_dist.ave_static_y == 4.
    assert diele_dist.ave_static_z == 1.5


def test_reciprocal_static(diele_dist):
    """The returned complex array contains ``y(0), y(1),..., y(n-1)``, where
       ``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``."""
    assert_almost_equal(diele_dist.reciprocal_static[0], [8.-0.j, -6.-0.j])


def test_dielectric_dist_to_plot(diele_dist):
    diele_dist.to_plot(plt.gca())
    plt.show()
