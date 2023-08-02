# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest
from matplotlib import pyplot as plt

from pydefect_2d.potential.grids import Grid
from pydefect_2d.potential.one_d_potential import OneDPotential, ExtremaDist, \
    OneDPotDiff


def test_1d_pot_vac_extremum_pot_pt():
    grid = Grid(10., 4)
    fp_1d = OneDPotential(charge_state=1, grid=grid,
                          potential=np.array([0.0, -0.1, 0.0, 0.1]))
    assert fp_1d.vac_extremum_pot_pt == 0.333


@pytest.fixture
def extrema_dist():
    return ExtremaDist(extrema=[0.1, 0.2, 0.3, 0.4],
                       gaussian_pos=[0.0, 0.1, 0.2, 0.3])


def test_extrema_dist_plot(extrema_dist):
    ax = plt.gca()
    extrema_dist.to_plot(ax)
    plt.show()


def test_one_d_pot_diff():
    one_d_pot_diff = OneDPotDiff(grid=Grid(10., 10),
                                 pot_1=np.linspace(0, 1, 10, endpoint=False),
                                 pot_2=np.linspace(0, 2, 10, endpoint=False),
                                 gauss_pos=0.5)
    assert one_d_pot_diff.vac_pot_gradient == 0.1



def test_extrema_dist_gauss_pos_from_extremum(extrema_dist):
    assert extrema_dist.gauss_pos_from_extremum(0.2) == 1.0











