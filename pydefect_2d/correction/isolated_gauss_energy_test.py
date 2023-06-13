# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest
from matplotlib import pyplot as plt
from numpy import sqrt, pi
from numpy.testing import assert_array_almost_equal
from scipy.constants import epsilon_0, elementary_charge, angstrom

from pydefect_2d.correction.isolated_gauss_energy import GaussEnergy
from pydefect_2d.potential.grids import Grid


sigma = 0.5


@pytest.fixture
def gauss_energy():
    return GaussEnergy(charge=1,
                       sigma=sigma,
                       z_grid=Grid(base_length=10.0, base_num_grid=10),
                       epsilon_z=[1.0]*10,
                       epsilon_xy=[1.0]*10,
                       z0=0.0,
                       k_max=10.,
                       k_mesh_dist=0.1)


def test_properties(gauss_energy: GaussEnergy):
    assert_array_almost_equal(gauss_energy.ks, [2., 4., 6., 8., 10.])
    assert gauss_energy.num_z_grid == 10
    assert gauss_energy.L == 10.0
    assert gauss_energy.Gs == [0.0, 0.6283185307179586, 1.2566370614359172, 1.8849555921538759, 2.5132741228718345, 3.141592653589793, 3.7699111843077517, 4.39822971502571, 5.026548245743669, 5.654866776461628]
    assert gauss_energy.rec_epsilon_z[0] == 10.-0.j
    assert gauss_energy.rec_epsilon_xy[0] == 10.-0.j


def test_gaussian_energy(gauss_energy: GaussEnergy):
    # epsilon_0 in C2 N-1 m-2
    expected = 1 / (8*pi**1.5*sigma) / epsilon_0 * elementary_charge / angstrom
    print(expected)
    print(gauss_energy.U)
    gauss_energy.to_plot(plt)
    plt.show()