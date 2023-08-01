# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest
from matplotlib import pyplot as plt

from pydefect_2d.potential.grids import Grid
from pydefect_2d.potential.one_d_potential import OneDimPotential, ExtremaDist


def test_1d_pot_vac_extremum_pot_pt():
    grid = Grid(10., 4)
    fp_1d = OneDimPotential(charge_state=1, grid=grid, potential=[0.0, -0.1, 0.0, 0.1])
    assert fp_1d.vac_extremum_pot_pt == 3.33


@pytest.fixture
def extrema_dist():
    return ExtremaDist(extrema=[0.1, 0.2, 0.3, 0.4],
                       gaussian_pos=[1.0, 1.1, 1.2, 1.3])


def test_extrema_dist_plot(extrema_dist):
    ax = plt.gca()
    extrema_dist.to_plot(ax)
    plt.show()


def test_extrema_dist_gauss_pos_from_extremum(extrema_dist):
    assert extrema_dist.gauss_pos_from_extremum(0.2) == 1.0











