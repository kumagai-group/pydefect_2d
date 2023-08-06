# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from functools import cached_property
from math import pi, exp

import numpy as np
from scipy.constants import epsilon_0, elementary_charge, angstrom
from scipy.fftpack import fft, ifft
from scipy.linalg import solve

from pydefect_2d.potential.dielectric_distribution import DielectricConstDist
from pydefect_2d.potential.grids import Grid
from pydefect_2d.potential.one_d_potential import OneDPotential, \
    Gauss1DPotential


@dataclass
class OneDGaussChargeModel:
    """Gauss charge model with 1|e| under periodic boundary condition. """
    grid: Grid
    sigma: float
    gauss_pos_in_frac: float  # in fractional coord. x=y=0
    surface: float = None  # in Å^2
    periodic_charges: np.array = None

    def __post_init__(self):
        if self.periodic_charges is None:
            self.periodic_charges = self._make_periodic_gauss_charge_profile

    @property
    def _surface(self):
        return self.surface or 1.0

    @property
    def _make_periodic_gauss_charge_profile(self):
        coefficient = 1 / self.sigma / (2 * pi) ** 0.5 / self._surface
        gauss = np.zeros(self.grid.num_grid)
        for nz, lz in enumerate(self.grid.grid_points):
            gauss[nz] = exp(-self._min_z2(lz) / (2 * self.sigma ** 2))

        return coefficient * gauss

    def _min_z2(self, lz):
        return min(
            [abs(lz - self.grid.length * (i + self.gauss_pos_in_frac))
             for i in range(-1, 2)]
        ) ** 2


@dataclass
class Calc1DPotential:
    diele_dist: DielectricConstDist  # [epsilon_x, epsilon_y, epsilon_z] along z
    one_d_gauss_charge_model: OneDGaussChargeModel  # assume orthogonal system

    @property
    def grid(self):
        return self.one_d_gauss_charge_model.grid

    @property
    def _rec_charge(self):
        result = fft(self.one_d_gauss_charge_model.periodic_charges)
        result[0] = 0.0
        return result

    def _solve_poisson_eq(self):
        z_num_grid = self.one_d_gauss_charge_model.grid.num_grid
        z_rec_e = self.diele_dist.reciprocal_static_z

        factors = []
        Gzs = self.one_d_gauss_charge_model.grid.Gs
        for i_gz, gz in enumerate(Gzs):
            inv_rho_by_mz = [z_rec_e[i_gz - i_gz_prime] * gz * gz_prime
                             for i_gz_prime, gz_prime in enumerate(Gzs)]
            if i_gz == 0:
                # To avoid a singular error, any non-zero value needs to be set.
                inv_rho_by_mz[0] = 1.0
            factors.append(inv_rho_by_mz)
        factors = np.array(factors)

        inv_pot_by_mz = solve(factors, self._rec_charge * z_num_grid, assume_a="her")
        inv_pot_by_mz[0] = 0.0
        return inv_pot_by_mz

    @cached_property
    def reciprocal_potential(self):
        inv_pot = self._solve_poisson_eq()
        return inv_pot / epsilon_0 * elementary_charge / angstrom

    @cached_property
    def potential(self):
        real = np.real(ifft(self.reciprocal_potential))
        return Gauss1DPotential(self.one_d_gauss_charge_model.grid,
                                real,
                                self.one_d_gauss_charge_model.gauss_pos_in_frac)
