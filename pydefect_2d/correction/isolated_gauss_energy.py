# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import multiprocessing as multi
from dataclasses import dataclass
from functools import cached_property
from multiprocessing import Pool
from typing import List

import numpy as np
from numpy import cos, exp, linspace
from scipy import integrate
from scipy.constants import pi, epsilon_0, elementary_charge, angstrom
from scipy.fft import fft
from tqdm import tqdm

from pydefect_2d.potential.grids import Grid


@dataclass
class GaussEnergy:
    charge: int
    sigma: float
    z_grid: Grid
    epsilon_z: List[float]
    epsilon_xy: List[float]
    z0: float
    k_max: float
    k_mesh_dist: float
    multiprocess: bool = True
    _U_ks: np.ndarray = None

    @property
    def ks(self):
        mesh_num = int(self.k_max / self.k_mesh_dist) + 1
        result = linspace(0, self.k_max, mesh_num, endpoint=True)
        return result[1:]

    @property
    def num_z_grid(self):
        return self.z_grid.num_grid

    @property
    def L(self):
        return self.z_grid.length

    @cached_property
    def Gs(self):
        return [2 * pi * i / self.L for i in range(self.num_z_grid)]

    @cached_property
    def rec_epsilon_z(self) -> List[float]:
        return fft(self.epsilon_z)

    @cached_property
    def rec_epsilon_xy(self) -> List[float]:
        return fft(self.epsilon_xy)

    def K_G(self, G, k):
        return self.L * (k**2+G**2) / (1-exp(-k*self.L/2)*cos(G*self.L/2))

    def D_GG(self, i, j, G_i, G_j, k) -> float:
        result = self.K_G(G_i, k) if G_i == G_j else 0.0
        result += (self.L * self.rec_epsilon_z[i - j] * G_i * G_j
                   + self.rec_epsilon_xy[i - i] * k ** 2)
        return result

    def D(self, k) -> np.array:
        result = np.zeros(shape=[self.num_z_grid, self.num_z_grid])
        for i, G_i in enumerate(self.Gs):
            for j, G_j in enumerate(self.Gs):
                i_j = self.D_GG(i, j, G_i, G_j, k)
                result[i, j] += i_j.real
                result[j, i] += i_j.real
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
    def U_ks(self):
        if self._U_ks is None:
            if self.multiprocess:
                p = Pool(multi.cpu_count())
                self._U_ks = np.array(p.map(self.U_k, tqdm(self.ks)))
            else:
                self._U_ks = np.array([self.U_k(k) for k in self.ks])

        return self._U_ks / epsilon_0 * elementary_charge / angstrom

    @cached_property
    def k_exps(self):
        return [k*exp(-k**2*self.sigma**2) for k in self.ks]

    @cached_property
    def Us(self):
        return [k_exp * uk for k_exp, uk in zip(self.k_exps, self.U_ks)]

    @cached_property
    def U(self):
        y_wo_zero = [k_exp * uk for k_exp, uk in zip(self.k_exps, self.U_ks)]
        x, y = np.insert(self.ks, 0, 0.0), np.insert(y_wo_zero, 0, 0.0)
        return integrate.trapz(y, x)

    def to_plot(self, plt):
        fig, axs = plt.subplots(3, 1, tight_layout=True, sharex=True)
        fig.supxlabel('k [Å$^{-1}$]')
        for ax, vals in zip(axs, [self.k_exps, self.U_ks, self.Us]):
            ax.set_yscale('log')
            ax.plot(self.ks, vals)
