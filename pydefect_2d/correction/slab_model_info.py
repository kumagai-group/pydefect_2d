# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import itertools
import multiprocessing as multi
from dataclasses import dataclass
from functools import cache, cached_property
from itertools import product
from math import pi, exp
from multiprocessing import Pool
from typing import List, Optional

import numpy as np
from monty.json import MSONable
from numpy import linspace
from scipy.constants import epsilon_0, elementary_charge, angstrom
from scipy.fft import fftn, fft, ifftn
from tqdm import tqdm
from vise.util.mix_in import ToJsonFileMixIn


@dataclass
class SlabGaussModel(MSONable, ToJsonFileMixIn):
    lattice_constants: List[float]  # assume orthogonal system
    epsilon: List[List[float]]  # [epsilon_x, epsilon_y, epsilon_z]
    charge: float
    std_dev: float
    defect_z_pos: float  # in fractional coord. x=y=0
    potential: Optional[np.array] = None
    multiprocess: bool = True

    @property
    def num_grids(self):
        return \
            [len(self.epsilon[0]), len(self.epsilon[1]), len(self.epsilon[2])]

    @cached_property
    def grids(self):
        return [linspace(0, lat, grids, False)
                for lat, grids in zip(self.lattice_constants, self.num_grids)]

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
        coefficient = self.charge / self.std_dev ** 3 / (2 * pi) ** 1.5
        gauss = np.zeros(self.num_grids, dtype=np.complex_)
        for ix, iy, iz in itertools.product(range(self.num_grids[0]),
                                            range(self.num_grids[1]),
                                            range(self.num_grids[2])):
            x2, y2 = self.grids[0][ix] ** 2, self.grids[1][iy] ** 2
            z2 = (self.grids[2][iz] - self.defect_z_pos) ** 2
            gauss[ix, iy, iz] = exp(-(x2 + y2 + z2) / (2 * self.std_dev ** 2))
        return coefficient * gauss

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
                collected_data = p.map(self._solve_poisson_eq, tqdm(grids))
        else:
            collected_data = [self._solve_poisson_eq(g) for g in grids]

        for d in collected_data:
            result[d[0], d[1], :] = d[2]

        result[0, 0, 0] = 0
        return result / epsilon_0 * elementary_charge / angstrom

    @cached_property
    def real_potential(self):
        if self.potential is None:
            self.potential = ifftn(self.reciprocal_potential)
        return self.potential

    @property
    def volume(self):
        return np.prod(self.lattice_constants)

    @property
    def xy_ave_potential(self):
        return np.real(self.real_potential.mean(axis=(0, 1)))

    @property
    def xy_averaged_charge(self):
        return np.real(self.real_charge.mean(axis=(0, 1)))

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
