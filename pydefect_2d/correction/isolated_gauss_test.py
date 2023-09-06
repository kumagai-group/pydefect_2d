# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pathlib import Path

import pytest
from monty.serialization import loadfn
from scipy.fft import fft

from pydefect_2d.correction.isolated_gauss import IsolatedGaussEnergy, \
    CalcIsolatedGaussEnergy

sigma = 1.0


path = Path(__file__).parent


@pytest.fixture
def gauss_energy():
    gauss = loadfn(path / "gauss_charge_model_0.500.json")
    diele = loadfn(path / "dielectric_const_dist.json")

    return CalcIsolatedGaussEnergy(gauss_charge_model=gauss,
                                   diele_const_dist=diele,
                                   k_max=1,
                                   k_mesh_dist=0.05,
                                   multiprocess=True).isolated_gauss_energy


def test_isolated_gaussian_energy(gauss_energy: IsolatedGaussEnergy):
    print(gauss_energy.self_energy)


# def test_gaussian_energy(gauss_energy: IsolatedGaussEnergy):
    # epsilon_0 in C2 N-1 m-2
    # expected = 1 / (8*pi**1.5*sigma) / epsilon_0 * elementary_charge / angstrom
    # print(expected)
    # print(gauss_energy.self_energy)
    # print(expected / gauss_energy.self_energy)
    # print(gauss_energy.U_k(0))
    # print(gauss_energy.U_k(0.1))
    # print(gauss_energy.U_k(0.2))
    # gauss_energy.to_plot(plt)
    # plt.show()


def test_():
    f = [0, 1, 2, 3, 2, 1]
    f = [0, 1, 2, 3, 3, 1]
    f = [2, 3, 2, 1, 0, 1]
    print(fft(f))


