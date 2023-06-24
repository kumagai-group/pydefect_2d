# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pydefect_2d.potential.slab_model_info import SlabModel


class ProfilePlotter:

    def __init__(self,
                 plt,
                 slab_model: SlabModel):
        self.plt = plt
        self.z_grid_points = slab_model.grids.z_grid_points
        self.charge = slab_model.gauss_charge_model.xy_integrated_charge
        self.epsilon = slab_model.epsilon.static

        self.fp_potential = None
        if slab_model.fp_potential:
            self.fp_potential = slab_model.fp_potential

        self.potential = slab_model.gauss_charge_potential.xy_ave_potential
        _, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, sharex="all")

        self._plot_potential()
        self._plot_charge()
        self._plot_epsilon()

        plt.subplots_adjust(hspace=.0)
        plt.xlabel("Distance (Å)")

    def _plot_charge(self):
        self.ax1.set_ylabel("Charge (|e|/Å)")
        self.ax1.plot(self.z_grid_points, self.charge,
                      label="charge", color="black")

    def _plot_epsilon(self):
        self.ax2.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
        for e, direction in zip(self.epsilon, ["x", "y", "z"]):
            self.ax2.plot(self.z_grid_points, e, label=direction)
        self.ax2.legend()

    def _plot_potential(self):
        self.ax3.set_ylabel("Potential energy (eV)")
        self.ax3.plot(self.z_grid_points, self.potential,
                      label="Gauss model", color="red")
        if self.fp_potential:
            self.ax3.plot(self.fp_potential.grid.grid_points,
                          self.fp_potential.potential,
                          label="FP", color="blue")
            self.ax3.plot(self.z_grid_points, self._diff_potential,
                          label="diff", color="green", linestyle=":")
        self.ax3.legend()

    @property
    def _diff_potential(self):
        fp_pot = self.fp_potential.interpol_pot_func(self.z_grid_points)
        return fp_pot - self.potential