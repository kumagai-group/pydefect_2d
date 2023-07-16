# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.constants import epsilon_0, elementary_charge, angstrom
from scipy.fftpack import ifft, fft

from pydefect_2d.potential.epsilon_distribution import DielectricConstDist
from pydefect_2d.potential.slab_model_info import GaussChargeModel, \
    FP1dPotential


@dataclass
class CalcOneDimGaussChargePotential:
    epsilon: DielectricConstDist  # [epsilon_x, epsilon_y, epsilon_z] along z
    gauss_charge_model: GaussChargeModel  # assume orthogonal system

    def __post_init__(self):
        try:
            assert self.epsilon.grid == self.gauss_charge_model.grids.z_grid
        except AssertionError:
            e_z_gird = self.epsilon.grid
            g_z_grid = self.gauss_charge_model.grids.z_grid

            print(f"epsilon z lattice length {e_z_gird.length}")
            print(f"epsilon num grid {e_z_gird.num_grid}")
            print(f"gauss model lattice length {g_z_grid.length}")
            print(f"gauss model num grid {g_z_grid.num_grid}")
            raise
        self.Ga2s = self.gauss_charge_model.grids.xy_grids.Ga2
        self.Gb2s = self.gauss_charge_model.grids.xy_grids.Gb2

    @property
    def z_grid(self):
        return self.gauss_charge_model.grids.z_grid

    @property
    def _rec_charge(self):
        return fft(self.gauss_charge_model.xy_average_charge)

    def _solve_poisson_eq(self):
        z_num_grid = self.gauss_charge_model.grids.z_grid.num_grid
        x_rec_e, y_rec_e, z_rec_e = self.epsilon.reciprocal_static

        factors = []
        Gzs = self.gauss_charge_model.grids.z_grid.Gs
        for i_gz, gz in enumerate(Gzs):
            inv_rho_by_mz = [z_rec_e[i_gz - i_gz_prime] * gz * gz_prime
                             for i_gz_prime, gz_prime in enumerate(Gzs)]
            if i_gz == 0:
                # To avoid a singular error, any non-zero value needs to be set.
                inv_rho_by_mz[0] = 1.0
            factors.append(inv_rho_by_mz)
        factors = np.array(factors)
        inv_pot_by_mz = np.linalg.solve(factors, self._rec_charge * z_num_grid)
        inv_pot_by_mz[0] = 0.0
        return inv_pot_by_mz

    @cached_property
    def reciprocal_potential(self):
        return self._solve_poisson_eq() / epsilon_0 * elementary_charge / angstrom

    @cached_property
    def potential(self):
        real = ifft(self.reciprocal_potential)
        return FP1dPotential(self.gauss_charge_model.grids.z_grid, real.tolist())
