# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from matplotlib import pyplot as plt

from pydefect_2d.correction.gaussian_energy import make_gaussian_energy, \
    GaussianEnergy, make_gaussian_energies, GaussianEnergies


def test_gaussian_energies():
    gauss_energies = GaussianEnergies(
        [GaussianEnergy(z=0.37,
                        isolated_energy=1.0352530163235767,
                        periodic_energy=0.5855912156941347)])
    gauss_energies.to_plot(plt.gca())
    plt.show()


def test_make_gaussian_energy(test_files):
    corr_dir = test_files / "correction"
    actual = make_gaussian_energy(corr_dir, z=0.37)
    expected = GaussianEnergy(z=0.37,
                              isolated_energy=1.0352530163235767,
                              periodic_energy=0.5855912156941347)
    assert actual == expected


def test_make_gaussian_energies(test_files):
    corr_dir = test_files / "correction"
    actual = make_gaussian_energies(corr_dir,
                                    z_range=[0.35, 0.38],
                                    inv_center=0.5)
    expected = GaussianEnergies([GaussianEnergy(z=0.37,
                                                isolated_energy=1.0352530163235767,
                                                periodic_energy=0.5855912156941347)])
    assert actual == expected
