# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from abc import ABC
from dataclasses import dataclass
from functools import cached_property
from typing import List

import numpy as np
from matplotlib.axes import Axes
from monty.json import MSONable
from scipy.constants import epsilon_0, elementary_charge, angstrom
from scipy.fftpack import fft, ifft
from scipy.interpolate import interp1d
from scipy.linalg import solve
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.one_d.charge import OneDGaussChargeModel

from pydefect_2d.three_d.grids import Grid


@dataclass
class OneDPotential(MSONable, ToJsonFileMixIn, ABC):
    """Abstract class for 1D potential profile"""
    grid: Grid
    potential: np.ndarray
    gauss_pos: float = None  # in fractional coordinate

    @property
    def grid_points(self):
        return self.grid.grid_points(end_point=True)

    @property
    def potential_w_end(self):
        return np.append(self.potential, self.potential[0])

    @cached_property
    def potential_func(self):
        return interp1d(self.grid_points, self.potential_w_end)

    def to_plot(self, ax, label=None):
        ax.set_xlabel("Distance (Å)")
        ax.set_ylabel("Potential energy (V)")
        ax.plot(self.grid_points, self.potential_w_end, label=label)


class OneDFpPotential(OneDPotential):
    pass


class OneDGaussPotential(OneDPotential):
    pass


@dataclass
class OneDPotDiff(MSONable, ToJsonFileMixIn):
    """Potential difference used for determining gaussian position"""
    fp_pot: OneDFpPotential
    gauss_pot: OneDGaussPotential

    def __post_init__(self):
        assert self.fp_pot.grid.length == self.gauss_pot.grid.length

    @property
    def fp_grid_points(self):
        return self.fp_pot.grid.grid_points(end_point=False)

    @property
    def potential_diff_gradient(self):
        pos = self.gauss_pot.gauss_pos * self.gauss_pot.grid.length
        idx, z = self.fp_pot.grid.farthest_grid_point(pos)
        idx_m1, idx_p1 = idx - 1, (idx + 1) % len(self.fp_grid_points)

        z_m1 = self.fp_grid_points[idx_m1]
        z_p1 = self.fp_grid_points[idx_p1]

        diff1 = (self.gauss_pot.potential_func(z_m1)
                 - self.fp_pot.potential[idx_m1])
        diff2 = (self.gauss_pot.potential_func(z_p1)
                 - self.fp_pot.potential[idx_p1])
        return (diff2 - diff1) / (self.fp_pot.grid.mesh_dist * 2)


@dataclass
class PotDiffGradients(MSONable, ToJsonFileMixIn):
    gradients: List[float]
    gauss_positions: List[float]  # in fractional coordinate

    def to_plot(self, ax: Axes):
        ax.set_xlabel("Gauss charge position (Å)")
        ax.set_ylabel("Gradient (V/Å)")
        ax.plot(self.gauss_positions, self.gradients, color="blue")
        ax.scatter(self.gauss_positions, self.gradients, color="blue")
        ax.axhline(y=0, linestyle="--")

    def gauss_pos_w_min_grad(self):
        idx = np.argmin(abs(np.array(self.gradients)))
        return round(self.gauss_positions[idx], 5)


@dataclass
class Calc1DPotential:
    diele_dist: DielectricConstDist  # [epsilon_x, epsilon_y, epsilon_z] along z
    one_d_gauss_charge_model: OneDGaussChargeModel  # assume orthogonal system

    @property
    def _reciprocal_charge(self):
        result = fft(self.one_d_gauss_charge_model.periodic_charges)
        result[0] = 0.0
        return result

    @property
    def z_rec_e(self):
        return self.diele_dist.reciprocal_static_z

    def _solve_poisson_eq(self):
        z_num_grid = self.one_d_gauss_charge_model.grid.num_grid

        factors = []
        Gzs = self.one_d_gauss_charge_model.grid.Gs
        for i_gz, gz in enumerate(Gzs):
            inv_rho_by_mz = [self.z_rec_e[i_gz - i_gz_prime] * gz * gz_prime
                             for i_gz_prime, gz_prime in enumerate(Gzs)]
            if i_gz == 0:
                # To avoid a singular error, any non-zero value needs to be set.
                inv_rho_by_mz[0] = 1.0
            factors.append(inv_rho_by_mz)
        factors = np.array(factors)

        inv_pot_by_mz = solve(factors,
                              self._reciprocal_charge * z_num_grid,
                              assume_a="her")
        inv_pot_by_mz[0] = 0.0
        return inv_pot_by_mz

    @cached_property
    def reciprocal_potential(self):
        inv_pot = self._solve_poisson_eq()
        return inv_pot / epsilon_0 * elementary_charge / angstrom

    @cached_property
    def potential(self):
        real = np.real(ifft(self.reciprocal_potential))
        return OneDGaussPotential(self.one_d_gauss_charge_model.grid,
                                  real,
                                  self.one_d_gauss_charge_model.gauss_pos_in_frac)
