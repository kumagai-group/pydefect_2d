# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest
from matplotlib import pyplot as plt

from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.dielectric.distribution import ManualDist
from pydefect_2d.one_d.charge import OneDGaussChargeModel

from pydefect_2d.three_d.grids import Grid
from pydefect_2d.one_d.potential import OneDPotDiff, \
    OneDGaussPotential, OneDFpPotential, PotDiffGradients, Calc1DPotential

grid = Grid(10., 4)


@pytest.fixture
def fp_pot():
    return OneDFpPotential(grid, potential=np.array([0.2, 0.4, 0.6, 0.8]))


@pytest.fixture
def gauss_pot():
    # gauss pos locate at grid index 3.
    return OneDGaussPotential(
        grid, np.array([0.0, -0.1, 0.0, 0.1]), gauss_pos=0.5)


def test_one_d_pot_diff(fp_pot, gauss_pot):
    one_d_pot_diff = OneDPotDiff(fp_pot, gauss_pot)
    actual = round(one_d_pot_diff.potential_diff_gradient, 2)
    expected = ((0.1 * 2 - 0.4) - (-0.1 * 2 - 0.2)) / (2 * 2.5)
    assert actual == expected


def test_pot_diff_gradients():
    pot_diff_gradients = PotDiffGradients(gradients=[0.9, -0.1, -1.3],
                                          gauss_positions=[0.4, 0.5, 0.6])
    assert pot_diff_gradients.gauss_pos_w_min_grad() == 0.5
    pot_diff_gradients.to_plot(plt.gca())
    plt.show()


def test_calc_1d_potential():
    n_grid = 100
    grid = Grid(10, n_grid)
    dist = ManualDist.from_grid(grid, np.array([1.0]*n_grid))
    charge_model = OneDGaussChargeModel(grid=grid,
                                        gauss_pos_in_frac=0.5,
                                        std_dev=0.1,
                                        surface=100.)
    diele_dist = DielectricConstDist([1.]*3, [0.]*3, dist)

    calc_1_potential = Calc1DPotential(diele_dist, charge_model)
    calc_1_potential.potential.to_plot(plt.gca())
    plt.show()
