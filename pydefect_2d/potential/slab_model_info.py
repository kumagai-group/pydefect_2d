# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import itertools
import multiprocessing as multi
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from math import pi, exp
from multiprocessing import Pool
from typing import List, Optional

import numpy as np
from scipy.constants import epsilon_0, elementary_charge, angstrom
from scipy.fft import fftn, fft, ifftn
from tqdm import tqdm

from pydefect_2d.potential.grids import Grids


@dataclass
class SlabGaussModel(Grids):
    epsilon: List[List[float]]  # [epsilon_x, epsilon_y, epsilon_z] along z
    charge: float
    sigma: float
    defect_z_pos: float  # in fractional coord. x=y=0
    charge_profile: Optional[np.array] = None
    potential_profile: Optional[np.array] = None
    multiprocess: bool = True

    @cached_property
    def Gs(self):
        # reduced zone
        result = []
        for num_grids, lat in zip(self.num_grids, self.lattice_constants):
#            result.append(2 * pi * np.array(range(num_grids)) / lat)
            igs = np.array(range(num_grids))
            middle_x = int(num_grids / 2) + 1
            igs[middle_x:] = igs[1:middle_x-1][::-1]
            result.append(2 * pi * igs / lat)

        return np.array(result)

    @cached_property
    def reciprocal_epsilon(self):
        return [fft(e) for e in self.epsilon]

    @cached_property
    def real_charge(self):
        if self.charge_profile is None:
            self._calc_charge_profile()
        return self.charge_profile

    def _calc_charge_profile(self):
        coefficient = self.charge / self.sigma ** 3 / (2 * pi) ** 1.5
        gauss = np.zeros(self.num_grids)
        lx, ly, lz = self.lattice_constants
        for ix, iy, iz in itertools.product(range(self.num_grids[0]),
                                            range(self.num_grids[1]),
                                            range(self.num_grids[2])):
            x2 = np.minimum(self.grids[0][ix] ** 2,
                            (lx - self.grids[0][ix]) ** 2)
            y2 = np.minimum(self.grids[0][iy] ** 2,
                            (ly - self.grids[0][iy]) ** 2)
            dz = abs(self.grids[2][iz] - self.defect_z_pos)
            z2 = np.minimum(dz ** 2, (lz - dz) ** 2)
            gauss[ix, iy, iz] = exp(-(x2 + y2 + z2) / (2 * self.sigma ** 2))
        self.charge_profile = coefficient * gauss

    @cached_property
    def reciprocal_charge(self):
        result = fftn(self.real_charge)
        result[0, 0, 0] = 0  # introduce background charge
        return result

    def _solve_poisson_eq(self, xy_grid_idx):
        # at a given Gx and Gy.
        i_gx, i_gy = xy_grid_idx
        gx, gy = self.Gs[0][i_gx], self.Gs[0][i_gy]
        rec_chg = self.reciprocal_charge[i_gx, i_gy, :]

        factors = []
        for i_gz, gz in enumerate(self.Gs[2]):
            inv_rho_by_mz = \
                [self.reciprocal_epsilon[0][i_gz - i_gz_prime] * gx ** 2 +
                 self.reciprocal_epsilon[1][i_gz - i_gz_prime] * gy ** 2 +
                 self.reciprocal_epsilon[2][i_gz - i_gz_prime] * gz * gz_prime
                 for i_gz_prime, gz_prime in enumerate(self.Gs[2])]
            factors.append(inv_rho_by_mz)
        factors = np.array(factors)
        inv_pot_by_mz = np.linalg.solve(factors, rec_chg * self.num_grids[2])
        return i_gx, i_gy, inv_pot_by_mz

    @cached_property
    def reciprocal_potential(self):
        result = np.zeros(self.num_grids, dtype=np.complex_)
        grids = [[i_gx, i_gy] for i_gx, i_gy
                 in product(range(self.num_grids[0]), range(self.num_grids[1]))]
        grids.pop(0)  # pop i_gx == 0 and i_gy == 0:
        p = Pool(multi.cpu_count())

        if self.multiprocess:
            with p:
                collected_data = tqdm(p.map(self._solve_poisson_eq, tqdm(grids)))
        else:
            collected_data = [self._solve_poisson_eq(g) for g in grids]

        for d in collected_data:
            result[d[0], d[1], :] = d[2]

        result[0, 0, 0] = 0
        return result / epsilon_0 * elementary_charge / angstrom

    @cached_property
    def real_potential(self):
        if self.potential_profile is None:
            self.potential_profile = ifftn(self.reciprocal_potential)
        return self.potential_profile

    @property
    def volume(self):
        return np.prod(self.lattice_constants)

    @property
    def xy_ave_potential(self):
        return np.real(self.real_potential.mean(axis=(0, 1)))

    @property
    def xy_sum_charge(self):
        return np.real(self.real_charge.sum(axis=(0, 1)))

    @property
    def electrostatic_energy(self):
        return np.real((np.mean(self.real_potential * self.real_charge)
                        * self.volume / 2))

    @property
    def farthest_z_from_defect(self):
        result = self.defect_z_pos + self.lattice_constants[2] / 2
        if result > self.lattice_constants[2]:
            result -= self.lattice_constants[2]
        return result


