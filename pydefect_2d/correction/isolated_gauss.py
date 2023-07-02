# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import multiprocessing as multi
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations_with_replacement
from multiprocessing import Pool
from typing import List

import numpy as np
from monty.json import MSONable
from numpy import cos, exp, linspace
from scipy import integrate, e
from scipy.constants import pi, epsilon_0, elementary_charge, angstrom
from scipy.fft import fft
from tqdm import tqdm
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.potential.slab_model_info import GaussChargeModel


@dataclass
class IsolatedGaussEnergy(MSONable, ToJsonFileMixIn):
    gauss_charge_model: GaussChargeModel
    epsilon_z: List[float]
    k_max: float
    k_mesh_dist: float
    multiprocess: bool = True
    _U_ks: np.ndarray = None

    def __post_init__(self):
        self.U_ks

    @property
    def sigma(self):
        return self.gauss_charge_model.sigma

    @property
    def L(self):
        return self.gauss_charge_model.grids.z_length

    @property
    def epsilon_xy(self):
        return self.gauss_charge_model.epsilon_ave

    @property
    def z0(self):
        return self.gauss_charge_model.defect_z_pos_in_frac

    @property
    def num_grid(self):
        return len(self.epsilon_z)

    @property
    def ks(self):
        mesh_num = int(self.k_max / self.k_mesh_dist) + 1
        result = linspace(0, self.k_max, mesh_num, endpoint=True)
        return result[1:]

    @cached_property
    def Gs(self):
        # reduced zone.
        half = int(self.num_grid / 2)
        return ([2*pi*n/self.L for n in range(half)]
                + [-2*pi*n/self.L for n in range(half, 0, -1)])

    @cached_property
    def rec_delta_epsilon_z(self) -> List[float]:
        # Need to be compatible with Gs
        return fft(np.array(self.epsilon_z) - 1.0) / self.num_grid

    @cached_property
    def rec_delta_epsilon_xy(self) -> List[float]:
        return fft(np.array(self.epsilon_xy) - 1.0) / self.num_grid

    def inv_K_G(self, G, k):
        denominator = 1. - e**(-k*self.L/2) * cos(G*self.L/2)
        return self.L * (k**2+G**2) / denominator

    def D_GG(self, i, j, k):
        G_i, G_j = self.Gs[i], self.Gs[j]
        result = self.inv_K_G(G_i, k) if i == j else 0.0
        L, rec_de_z, rec_de_xy = \
            self.L, self.rec_delta_epsilon_z, self.rec_delta_epsilon_xy
        result += L * (rec_de_z[i-j] * G_i*G_j + rec_de_xy[i-j] * k**2)
        return result

    def D(self, k) -> np.array:
        result = np.zeros(shape=[self.num_grid, self.num_grid],dtype=complex)
        result[:] = np.nan
        for i, j in combinations_with_replacement(range(self.num_grid), 2):
            result[i, j] = result[j, i] = self.D_GG(i, j, k)
        return result

    def inv_D(self, k) -> np.array:
        return np.linalg.inv(self.D(k))

    def U_k(self, k):
        result = 0.
        inv_D = self.inv_D(k)
        for i, G_i in enumerate(self.Gs):
            for j, G_j in enumerate(self.Gs):
                exp_inner = 0.+1.j*(G_i-G_j)*self.z0 \
                            - (G_i**2+G_j**2)*self.sigma**2/2
                result += exp(exp_inner) * inv_D[i, j]
        return result

        # return sum([exp(-(G_i*self.sigma)**2) / (k**2+G_i**2) * (1. - e**(-k*self.L/2) * cos(G_i*self.L/2))
        #             for G_i in self.Gs]) / self.L
        # return sum([exp(-(G_i*self.sigma)**2) / (k**2+G_i**2)
        #               for G_i in self.Gs]) / self.L
        # return exp(self.sigma**2 * k**2) / k * erfc(self.sigma * k) / 2

    @cached_property
    def U_ks(self):
        if self._U_ks is None:
            if self.multiprocess:
                p = Pool(multi.cpu_count())
                self._U_ks = np.array(p.map(self.U_k, tqdm(self.ks)))
            else:
                self._U_ks = np.array([self.U_k(k) for k in self.ks])

        factor = 1 / epsilon_0 * elementary_charge / angstrom / (4*pi)
        return self._U_ks.real * factor

    @cached_property
    def k_exps(self):
        return [k*exp(-k**2*self.sigma**2) for k in self.ks]

    @cached_property
    def Us(self):
        return [k_exp * uk for k_exp, uk in zip(self.k_exps, self.U_ks)]

    @cached_property
    def self_energy(self):
        x, y = np.insert(self.ks, 0, 0.0), np.insert(self.Us, 0, 0.0)
        return integrate.trapz(y, x)

    def to_plot(self, plt):
        fig, axs = plt.subplots(3, 1, tight_layout=True, sharex=True)
        fig.supxlabel('k [Å$^{-1}$]')
        for ax, vals in zip(axs, [self.k_exps, self.U_ks, self.Us]):
            ax.set_yscale('log')
            x = np.insert(self.ks, 0, 0.0)
            y = np.insert(vals, 0, 0.0)
            ax.plot(x, y)

    def __str__(self):
        return f"self energy: {self.self_energy:.5} eV"


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
