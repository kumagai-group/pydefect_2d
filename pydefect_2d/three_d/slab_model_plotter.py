# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from abc import ABC, abstractmethod


class SlabModelPlotAbs(ABC):

    @abstractmethod
    def gauss_charge_z_plot(self, ax):
        pass

    @abstractmethod
    def gauss_potential_z_plot(self, ax):
        pass

    @abstractmethod
    def fp_potential_plot(self, ax):
        pass

    @abstractmethod
    def epsilon_plot(self, ax):
        pass

    @property
    @abstractmethod
    def z_length(self):
        pass


class SlabModelPlotter:

    def __init__(self,
                 plt,
                 slab_model_plot: SlabModelPlotAbs):
        self.plt = plt
        self.slab_model_pot = slab_model_plot

        _, (self.ax1, self.ax3) = plt.subplots(2, 1, sharex="all")
        self.ax2 = self.ax1.twinx()
        self.ax3.set_xlabel("Distance (Å)")

        self._plot_charge()
        self._plot_epsilon()
        self._plot_potential()
        self._plot_fp_potential()

        plt.xlim(0.0, self.slab_model_pot.z_length)
        plt.subplots_adjust(hspace=.0)
        self.ax1.axhline(y=0, color="black", linestyle=":", linewidth=0.5)
        self.ax3.axhline(y=0, color="black", linestyle=":", linewidth=0.5)

    def _plot_charge(self):
        self.slab_model_pot.gauss_charge_z_plot(self.ax1)

    def _plot_epsilon(self):
        self.slab_model_pot.epsilon_plot(self.ax2)

    def _plot_fp_potential(self):
        self.slab_model_pot.fp_potential_plot(self.ax3)

    def _plot_potential(self):
        self.slab_model_pot.gauss_potential_z_plot(self.ax3)
