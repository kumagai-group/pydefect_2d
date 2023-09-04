# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pydefect_2d.potential.slab_model_info import SlabModel


class ProfilePlotter:

    def __init__(self,
                 plt,
                 slab_model: SlabModel):
        self.plt = plt
        self.slab_model = slab_model
        self.gauss_charge_model = slab_model.gauss_charge_model
        self.gauss_charge_potential = slab_model.gauss_charge_potential
        self.z_grid_points = slab_model.grids.z_grid.grid_points()
        self.epsilon = slab_model.diele_dist.static

        self.fp_potential = None
        if slab_model.fp_potential:
            self.fp_potential = slab_model.fp_potential

        self.potential = slab_model.xy_ave_pot
        _, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, sharex="all")

        self._plot_potential()
        self._plot_charge()
        self._plot_epsilon()

        plt.subplots_adjust(hspace=.0)
        plt.xlabel("Distance (Å)")

    def _plot_charge(self):
        self.gauss_charge_model.to_plot(self.ax1,
                                        charge=self.slab_model.charge_state)

    def _plot_epsilon(self):
        self.ax2.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
        for e, direction in zip(self.epsilon, ["x", "y", "z"]):
            self.ax2.plot(self.z_grid_points, e, label=direction)
        self.ax2.legend()

    def _plot_potential(self):
        if self.fp_potential:
            self.ax3.plot(self.fp_potential.grid.grid_points(),
                          self.fp_potential.potential,
                          label="FP", color="blue")
            self.ax3.plot(self.z_grid_points, self._diff_potential,
                          label="diff", color="green", linestyle=":")

        self.gauss_charge_potential.to_plot(self.ax3,
                                            charge=self.slab_model.charge_state)

    @property
    def _diff_potential(self):
        fp_pot = self.fp_potential.potential_func(self.z_grid_points)
        return fp_pot - self.potential
