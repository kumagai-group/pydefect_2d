# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from typing import List


class GaussianEnergy:
    z: float
    isolated_energy: float
    periodic_energy: float


class GaussianEnergies:
    gaussian_energies: List[GaussianEnergy]

    def to_plot(self, ax, charge=1):
        zs, corr = [], []
        for ge in self.gaussian_energies:
            zs.append(ge.z)
            corr.append((ge.isolated_energy - ge.periodic_energy) * charge)

        ax.set_ylabel("Correction energy (eV)")
        ax.plot(zs, corr, label="corr", color="black")

