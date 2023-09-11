# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from matplotlib import pyplot as plt

from pydefect_2d.correction.gauss_energy import make_gauss_energy, \
    GaussEnergy, make_gauss_energies, GaussEnergies


def test_gaussian_energies():
    gauss_energies = GaussEnergies(
        [GaussEnergy(z=0.37,
                     isolated_energy=1.0352530163235767,
                     periodic_energy=0.5855912156941347)])
    gauss_energies.to_plot(plt.gca())
    plt.show()


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
