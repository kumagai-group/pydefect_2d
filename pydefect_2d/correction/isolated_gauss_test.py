# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy import pi
from scipy.constants import epsilon_0, elementary_charge, angstrom

from pydefect_2d.correction.isolated_gauss import IsolatedGaussEnergy
from pydefect_2d.potential.grids import Grids, Grid
from pydefect_2d.potential.slab_model_info import GaussChargeModel

sigma = 1.0


@pytest.fixture
def gauss_energy():
    n_grid = 50
    grids = Grids([Grid(length=100., num_grid=n_grid)] * 3)
    charge_model = GaussChargeModel(grids=grids,
                                    sigma=sigma,
                                    defect_z_pos=0.0,
                                    epsilon_x=np.array([1.0] * n_grid),
                                    epsilon_y=np.array([1.0] * n_grid))

    return IsolatedGaussEnergy(gauss_charge_model=charge_model,
                               epsilon_z=[1.0]*n_grid,
                               k_max=2.,
                               k_mesh_dist=0.01,
                               multiprocess=True)


def test_gaussian_energy(gauss_energy: IsolatedGaussEnergy):
    # epsilon_0 in C2 N-1 m-2
    expected = 1 / (8*pi**1.5*sigma) / epsilon_0 * elementary_charge / angstrom
    print(expected)
    print(gauss_energy.self_energy)
    print(expected / gauss_energy.self_energy)
    # print(gauss_energy.U_k(0))
    # print(gauss_energy.U_k(0.1))
    # print(gauss_energy.U_k(0.2))
    # gauss_energy.to_plot(plt)
    # plt.show()