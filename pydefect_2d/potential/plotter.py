# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from matplotlib import pyplot as plt

from pydefect_2d.potential.first_principles_potential import \
    FirstPrinciplesPotentialProfile
from pydefect_2d.potential.slab_model_info import SlabGaussModel


class ProfilePlotter:

    def __init__(self, model: SlabGaussModel):
        print(model.charge_profile)
        self.plt = plt
        self.z_grid = model.grids[2]
        self.charge = model.xy_sum_charge
        self.epsilon = model.epsilon

        if model.potential_profile is None:
            _, (self.ax1, self.ax2) = plt.subplots(2, 1, sharex="all")
        else:
            self.potential = model.xy_ave_potential
            _, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, sharex="all")
            self._plot_potential()

        self._plot_charge()
        self._plot_epsilon()
        plt.subplots_adjust(hspace=.0)
        plt.xlabel("Distance (Å)")

    def _plot_charge(self):
        self.ax1.set_ylabel("Charge (|e|/Å)")
        self.ax1.plot(self.z_grid, self.charge,
                      label="charge", color="black")

    def _plot_epsilon(self):
        self.ax2.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
        for e, direction in zip(self.epsilon, ["x", "y", "z"]):
            self.ax2.plot(self.z_grid, e, label=direction)
        self.ax2.legend()

    def _plot_potential(self):
        self.ax3.set_ylabel("Potential energy (eV)")
        self.ax3.plot(self.z_grid, self.potential,
                      label="potential", color="black")
