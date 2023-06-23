# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest
from matplotlib import pyplot as plt
from numpy import sqrt, pi
from numpy.testing import assert_array_almost_equal
from scipy.constants import epsilon_0, elementary_charge, angstrom

from pydefect_2d.correction.isolated_gauss_energy import GaussEnergy
from pydefect_2d.potential.grids import Grid
from pydefect_2d.potential.grids_test import grids
from pydefect_2d.potential.slab_model_info import SingleGaussChargeModel

sigma = 1.0


@pytest.fixture
def gauss_energy():
    Grid(base_length=1.0, base_num_grid=2, mul=2)
    charge_model = SingleGaussChargeModel(grids=grids)
    return GaussEnergy(charge=1,
                       sigma=sigma,
                       L=200.0,
                       epsilon_z=[1.0]*20,
                       epsilon_xy=[1.0]*20,
                       z0=0.0,
                       k_max=10.,
                       k_mesh_dist=0.001,
                       multiprocess=False)


def test_properties(gauss_energy: GaussEnergy):
    assert_array_almost_equal(gauss_energy.ks, [2., 4., 6., 8., 10.])
    assert gauss_energy.num_grid == 10
    assert gauss_energy.L == 10.0
    assert gauss_energy.Gs == [0.0, 0.6283185307179586, 1.2566370614359172, 1.8849555921538759, 2.5132741228718345, 3.141592653589793, 3.7699111843077517, 4.39822971502571, 5.026548245743669, 5.654866776461628]
    assert gauss_energy.rec_delta_epsilon_z[0] == 10. - 0.j
    assert gauss_energy.rec_delta_epsilon_xy[0] == 10. - 0.j


def test_gaussian_energy(gauss_energy: GaussEnergy):
    # epsilon_0 in C2 N-1 m-2
    expected = 1 / (8*pi**1.5*sigma) / epsilon_0 * elementary_charge / angstrom
    print(expected)
    print(gauss_energy.self_energy)
    gauss_energy.to_plot(plt)
    plt.show()