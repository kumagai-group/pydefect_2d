# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal

from pydefect_2d.potential.distribution import ManualDist
from pydefect_2d.potential.dielectric_distribution import \
    DielectricConstDist
from pydefect_2d.potential.grids import Grid


grid = Grid(20., 2)


@pytest.fixture
def dielectric_dist():
    return DielectricConstDist(ave_ele=[3., 3., 0.5],
                               ave_ion=[1., 1., 1.0],
                               dist=ManualDist.from_grid(grid, [0.0, 1.0]))


def test_dielectric_dist_properties(dielectric_dist: DielectricConstDist):
    assert dielectric_dist.grid_points == [0.0, 10.0]
    assert dielectric_dist.ave_static_x == 4.
    assert dielectric_dist.ave_static_y == 4.
    assert dielectric_dist.ave_static_z == 1.5


def test_reciprocal_static(dielectric_dist):
    """The returned complex array contains ``y(0), y(1),..., y(n-1)``, where
       ``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``."""
    assert_almost_equal(dielectric_dist.reciprocal_static[0], [8.-0.j, -6.-0.j])


def test_dielectric_dist_to_plot(dielectric_dist):
    dielectric_dist.to_plot(plt.gca())
    plt.show()
