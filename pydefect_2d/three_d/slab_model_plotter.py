# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

from pydefect_2d.three_d.slab_model import SlabModel
from pydefect_2d.util.utils import with_end_point


class SlabModelPlotter:

    def __init__(self,
                 plt,
                 slab_model: SlabModel):
        self.plt = plt
        self.slab_model = slab_model
        self.gauss_charge_model = slab_model.gauss_charge_model
        self.gauss_charge_potential = slab_model.gauss_charge_potential
        self.z_grid_points = slab_model.grids.z_grid.grid_points(end_point=True)
        self.epsilon = slab_model.diele_dist.static

        self.potential = slab_model.xy_ave_pot
        _, (self.ax1, self.ax3) = plt.subplots(2, 1, sharex="all")
        self.ax2 = self.ax1.twinx()
        self.ax3.set_xlabel("Distance (Å)")

        self._plot_charge()
        self._plot_epsilon()
        if slab_model.fp_potential:
            self._plot_fp_potential()
        self._plot_potential()

        plt.xlim(self.z_grid_points[0], self.z_grid_points[-1])
        plt.subplots_adjust(hspace=.0)
        self.ax1.axhline(y=0, color="black", linestyle=":", linewidth=0.5)
        self.ax3.axhline(y=0, color="black", linestyle=":", linewidth=0.5)

    def _plot_charge(self):
        self.gauss_charge_model.to_plot(
            self.ax1, charge=self.slab_model.charge_state)

    def _plot_epsilon(self):
        self.ax2.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
        for e, direction in zip(self.epsilon, ["x", "y", "z"]):
            self.ax2.plot(self.z_grid_points, with_end_point(e),
                          label=direction)
        self.ax2.legend()

    def _plot_fp_potential(self):
        fp_potential = self.slab_model.fp_potential
        self.ax3.plot(fp_potential.grid.grid_points(True),
                      with_end_point(fp_potential.potential),
                      label="FP", color="blue")

        fp_pot = fp_potential.potential_func(self.z_grid_points)
        diff_pot = fp_pot - with_end_point(self.potential)
        self.ax3.plot(self.z_grid_points, diff_pot,
                      label="diff", color="green", linestyle=":")

    def _plot_potential(self):
        self.gauss_charge_potential.to_plot(
            self.ax3, charge=self.slab_model.charge_state)
