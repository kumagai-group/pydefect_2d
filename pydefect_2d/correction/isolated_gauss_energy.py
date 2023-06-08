# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import multiprocessing as multi
from dataclasses import dataclass
from functools import cached_property
from multiprocessing import Pool
from typing import List

import numpy as np
from numpy import cos, exp, linspace
from scipy.constants import pi
from tqdm import tqdm

from pydefect_2d.potential.grids import Grid


@dataclass
class GaussEnergy:
    charge: int
    sigma: float
    grid: Grid
    epsilon: List[float]
    z0: List[float]
    k_max: float
    k_mesh_dist: int
    multiprocess: bool = True
    U_ks: np.ndarray = None

    @property
    def ks(self):
        return linspace(0, self.k_max, self.k_mesh_dist)

    @property
    def num_grid(self):
        return self.grid.num_grid

    @property
    def L(self):
        return self.grid.length

    @cached_property
    def Gs(self):
        return [2*pi*i / self.L for i in range(self.num_grid)]

    @cached_property
    def reciprocal_epsilon(self) -> List[float]:
        return [[1]]*self.num_grid

    def K_G(self, G, k):
        return self.L * (k**2+G**2) / (1-exp(-k*self.L/2)*cos(G*self.L/2))

    def D_GG(self, G1, G2, k) -> float:
        result = self.K_G(G1, k) if G1 == G2 else 0.0
        result += self.L * self.reciprocal_epsilon[G1 - G2]*(G1*G2+k**2)
        return result

    def D(self, k) -> np.array:
        result = np.zeros(shape=[self.num_grid, self.num_grid])
        for i, G_i in enumerate(self.Gs):
            for j, G_j in enumerate(self.Gs):
                i_j = self.D_GG(G_i, G_j, k)
                result[i, j] += i_j
                result[j, i] += i_j
        return result

    def inv_D(self, k) -> np.array:
        return np.linalg.inv(self.D(k))

    def U_k(self, k):
        result = 0.0
        inv_D = self.inv_D(k)
        for i, G_i in enumerate(self.Gs):
            for j, G_j in enumerate(self.Gs):
                result += cos((G_i - G_j)*self.z0) \
                          * exp(-(G_i**2+G_j**2) * self.sigma ** 2 / 2) \
                          * inv_D[i, j]
        return result

    @cached_property
    def _U_ks(self):
        if self.U_ks is not None:
            return self.U_ks

        if self.multiprocess:
            p = Pool(multi.cpu_count())
            return p.map(self.U_k, tqdm(self.ks))
        else:
            return [self.U_k(k) for k in self.ks]

#    def __str__(self):


