# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
from numpy.testing import assert_array_almost_equal

from pydefect_2d.potential.grids import Grid
from pydefect_2d.potential.one_d_potential import OneDPotDiff, \
    Gauss1DPotential, Fp1DPotential


def test_1d_pot_vac_extremum_pot_pt():
    grid = Grid(10., 4)
    pot = Gauss1DPotential(grid=grid,
                           _potential=np.array([0.0, -0.1, 0.0, 0.1]),
                           gauss_pos=0.5,
                           charge_state=2)
    assert_array_almost_equal(pot.potential, np.array([0.0, -0.2, 0.0, 0.2]))


def test_one_d_pot_diff():
    grid = Grid(10., 4)
    fp = Fp1DPotential(grid,
                       _potential=np.array([0.1, 0.2, 0.3, 0.4]),
                       charge_state=2)
    gauss = Gauss1DPotential(grid,
                             np.array([0.0, -0.1, 0.0, 0.1]),
                             gauss_pos=0.5,  # idx = 3
                             charge_state=2)

    one_d_pot_diff = OneDPotDiff(fp, gauss)
    assert one_d_pot_diff.pot_diff_grad == ((-0.1*2-0.2) - (0.1*2-0.4)) / (2*2.5)



# @pytest.fixture
# def extrema_dist():
#     return ExtremaDist(extrema=[0.1, 0.2, 0.3, 0.4],
#                        gaussian_pos=[0.0, 0.1, 0.2, 0.3])


# def test_extrema_dist_plot(extrema_dist):
#     ax = plt.gca()
#     extrema_dist.to_plot(ax)
#     plt.show()



# def test_extrema_dist_gauss_pos_from_extremum(extrema_dist):
#     assert extrema_dist.gauss_pos_from_extremum(0.2) == 1.0











