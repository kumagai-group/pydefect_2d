# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest
from matplotlib import pyplot as plt

from pydefect_2d.correction.gauss_energy import make_gauss_energy, \
    GaussEnergy, make_gauss_energies, GaussEnergies


@pytest.fixture
def gauss_energies():
    return GaussEnergies(
    [
        GaussEnergy(z=0.35, isolated_energy=1.0, periodic_energy=0.5),
        GaussEnergy(z=0.45, isolated_energy=2.5, periodic_energy=1.0),
        GaussEnergy(z=0.55, isolated_energy=3.5, periodic_energy=1.5),
        GaussEnergy(z=0.65, isolated_energy=5.5, periodic_energy=1.5),
    ])


def test_gaussian_energies_plot(gauss_energies):
    gauss_energies.to_plot(plt.gca())
    plt.show()


def test_gaussian_energies_get_gauss_energy(gauss_energies):
    actual = gauss_energies.get_gauss_energy(z=0.40)
    expected = GaussEnergy(z=0.40, isolated_energy=1.9062500000000004,
                           periodic_energy=0.71875)
    assert actual == expected


def test_make_gaussian_energy(test_files):
    corr_dir = test_files / "correction"
    actual = make_gauss_energy(corr_dir, z=0.37)
    expected = GaussEnergy(z=0.37,
                           isolated_energy=1.0352530163235767,
                           periodic_energy=0.5855912156941347)
    assert actual == expected


def test_make_gaussian_energies(test_files):
    corr_dir = test_files / "correction"
    actual = make_gauss_energies(corr_dir,
                                 z_range=[0.35, 0.38])
    expected = GaussEnergies([GaussEnergy(z=0.37,
                                          isolated_energy=1.0352530163235767,
                                          periodic_energy=0.5855912156941347)])
    assert actual == expected
