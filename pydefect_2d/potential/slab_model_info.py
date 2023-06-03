# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import itertools
import multiprocessing as multi
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from math import pi, exp
from multiprocessing import Pool
from typing import List

import numpy as np
from monty.json import MSONable
from scipy.constants import epsilon_0, elementary_charge, angstrom
from scipy.fftpack import ifftn, fftn
from scipy.interpolate import interpolate
from tabulate import tabulate
from tqdm import tqdm
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.potential.make_epsilon_distribution import Grid, \
    EpsilonDistribution


@dataclass
class Grids(MSONable):
    grids: List[Grid]

    def __call__(self, *args, **kwargs):
        return self.grids

    @property
    def all_grid_points(self):
        return [g.grid_points for g in self.grids]

    @property
    def num_grid_points(self):
        return [grid.num_grid for grid in self.grids]

    @property
    def lengths(self):
        return [grid.length for grid in self.grids]

    @property
    def xy_area(self):
        return np.prod(self.lengths[:2])

    @property
    def z_length(self):
        return self.grids[2].length

    @property
    def z_grid_points(self):
        return self.grids[2].grid_points

    @property
    def volume(self):
        return np.prod(self.lengths)


@dataclass
class GaussChargeModel(MSONable, ToJsonFileMixIn):
    grids: Grids  # assume orthogonal system
    charge: float
    sigma: float
    defect_z_pos: float  # in fractional coord. x=y=0
    charges: np.array = None

    def __post_init__(self):
        if self.charges is None:
            self.charges = self._make_gauss_charge_profile

    @property
    def _make_gauss_charge_profile(self):
        coefficient = self.charge / self.sigma ** 3 / (2 * pi) ** 1.5

        x_pts, y_pts, z_pts = self.grids.all_grid_points
        nx, ny, nz = self.grids.num_grid_points
        lx, ly, lz = self.grids.lengths

        gauss = np.zeros([nx, ny, nz])

        for ix, iy, iz in itertools.product(range(nx), range(ny), range(nz)):
            x2 = np.minimum(x_pts[ix] ** 2, (lx - x_pts[ix]) ** 2)
            y2 = np.minimum(y_pts[iy] ** 2, (ly - y_pts[iy]) ** 2)
            dz = abs(z_pts[iz] - self.defect_z_pos)
            z2 = np.minimum(dz ** 2, (lz - dz) ** 2)
            gauss[ix, iy, iz] = exp(-(x2 + y2 + z2) / (2 * self.sigma ** 2))

        return coefficient * gauss

    @cached_property
    def reciprocal_charge(self):
        result = fftn(self.charges)
        result[0, 0, 0] = 0  # introduce background charge
        return result

    @cached_property
    def xy_integrated_charge(self):
        return np.real(self.charges.mean(axis=(0, 1))) * self.grids.xy_area

    @property
    def farthest_z_from_defect(self):
        result = self.defect_z_pos + self.grids.z_length / 2
        if result > self.grids.z_length:
            result -= self.grids.z_length
        return result


@dataclass
class Potential(MSONable, ToJsonFileMixIn):
    grids: Grids  # assume orthogonal system
    potential: np.array  # potential for positive charge

    @cached_property
    def xy_ave_potential(self):
        return np.real(self.potential.mean(axis=(0, 1)))


@dataclass
class SlabModel(MSONable, ToJsonFileMixIn):
    epsilon: EpsilonDistribution  # [epsilon_x, epsilon_y, epsilon_z] along z
    charge: GaussChargeModel
    potential: Potential

    def __post_init__(self):
        assert self.epsilon.grid == self.charge.grids()[2]
        assert self.charge.grids == self.potential.grids

    @property
    def grids(self) -> Grids:
        return self.charge.grids

    @cached_property
    def electrostatic_energy(self):
        return np.real((np.mean(self.potential.potential * self.charge.charges)
                        * self.charge.grids.volume / 2))

    def __str__(self):
        header = ["pos (Å)", "charge", "potential"]
        list_ = []
        for i, pos in enumerate(self.grids.z_length):
            list_.append([pos,
                          self.charge.xy_integrated_charge[i],
                          self.potential.xy_ave_potential[i]])

        result = [tabulate(list_, tablefmt="plain", headers=header)]

        charge_sum = self.charge.charges.mean() * self.grids.volume
        result.append(f"Charge sum (|e|): {charge_sum:.3}")
        result.append(f"Electrostatic energy (eV): "
                      f"{self.electrostatic_energy:.3}")
        return "\n".join(result)


@dataclass
class CalcPotential(MSONable, ToJsonFileMixIn):
    epsilon: EpsilonDistribution  # [epsilon_x, epsilon_y, epsilon_z] along z
    gauss_model: GaussChargeModel  # assume orthogonal system
    multiprocess: bool = True

    def __post_init__(self):
        try:
            assert self.epsilon.grid == self.gauss_model.grids()[2]
        except AssertionError:
            e_z_gird, g_z_grid = self.epsilon.grid, self.gauss_model.grids()[2]
            print(f"epsilon z lattice length {e_z_gird.length}")
            print(f"epsilon num grid {e_z_gird.num_grid}")
            print(f"gauss model lattice length {g_z_grid.length}")
            print(f"gauss model num grid {g_z_grid.num_grid}")
            raise

    @property
    def num_grids(self):
        return [g.num_grid for g in self.gauss_model.grids()]

    @property
    def lattice_constants(self):
        return [g.length for g in self.gauss_model.grids()]

    @cached_property
    def Gs(self):
        result = []
        for num_grids, lat in zip(self.num_grids, self.lattice_constants):
            igs = np.array(range(num_grids))
            middle_x = int(num_grids / 2) + 1
            igs[middle_x:] = igs[1:middle_x-1][::-1]  # reduced zone
            result.append(2 * pi * igs / lat)

        return np.array(result)

    def _solve_poisson_eq(self, xy_grid_idx):
        # at a given Gx and Gy.
        i_gx, i_gy = xy_grid_idx
        gx, gy = self.Gs[0][i_gx], self.Gs[0][i_gy]
        z_grid = self.num_grids[2]
        x_rec_e, y_rec_e, z_rec_e = self.epsilon.reciprocal_static
        rec_chg = self.gauss_model.reciprocal_charge[i_gx, i_gy, :]

        factors = []
        for i_gz, gz in enumerate(self.Gs[2]):
            inv_rho_by_mz = [x_rec_e[i_gz - i_gz_prime] * gx ** 2 +
                             y_rec_e[i_gz - i_gz_prime] * gy ** 2 +
                             z_rec_e[i_gz - i_gz_prime] * gz * gz_prime
                             for i_gz_prime, gz_prime in enumerate(self.Gs[2])]
            if i_gx == 0 and i_gy == 0 and i_gz == 0:
                inv_rho_by_mz[0] = 1.0
            factors.append(inv_rho_by_mz)
        factors = np.array(factors)
        inv_pot_by_mz = np.linalg.solve(factors, rec_chg * z_grid)
        return i_gx, i_gy, inv_pot_by_mz

    @cached_property
    def reciprocal_potential(self):
        x_grids, y_grids = self.num_grids[:2]

        result = np.zeros(self.num_grids, dtype=np.complex_)
        grids = [[i_gx, i_gy] for i_gx, i_gy
                 in product(range(x_grids), range(y_grids))]
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
    def potential(self):
        real = ifftn(self.reciprocal_potential)
        return Potential(self.gauss_model.grids, real)
#    def to_plot(self, plt):
#        ProfilePlotter(self, plt)


@dataclass
class FP1dPotential(MSONable, ToJsonFileMixIn):
    grid: Grid
    fp_xy_ave_potential: List[float]


class ProfilePlotter:

    def __init__(self,
                 plt,
                 slab_model: SlabModel,
                 fp_potential: FP1dPotential = None):
        self.plt = plt
        self.z_grid_points = slab_model.grids.z_grid_points
        self.charge = slab_model.charge.xy_integrated_charge
        self.epsilon = slab_model.epsilon.static

        self.fp_grid, self.fp_potential = None, None
        if fp_potential:
            self.fp_grid = fp_potential.grid.grid_points
            self.fp_potential = fp_potential.fp_xy_ave_potential

        self.potential = slab_model.potential.xy_ave_potential
        _, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, sharex="all")

        self._plot_potential()
        self._plot_charge()
        self._plot_epsilon()

        plt.subplots_adjust(hspace=.0)
        plt.xlabel("Distance (Å)")

    def _plot_charge(self):
        self.ax1.set_ylabel("Charge (|e|/Å)")
        self.ax1.plot(self.z_grid_points, self.charge,
                      label="charge", color="black")

    def _plot_epsilon(self):
        self.ax2.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
        for e, direction in zip(self.epsilon, ["x", "y", "z"]):
            self.ax2.plot(self.z_grid_points, e, label=direction)
        self.ax2.legend()

    def _plot_potential(self):
        self.ax3.set_ylabel("Potential energy (eV)")
        self.ax3.plot(self.z_grid_points, self.potential,
                      label="Gaussian model", color="red")
        if self.fp_potential:
            self.ax3.plot(self.fp_grid, self.fp_potential,
                          label="FP", color="blue")
            self.ax3.plot(self.z_grid_points, self._diff_potential,
                          label="diff", color="green", linestyle=":")
        self.ax3.legend()

    @property
    def _diff_potential(self):
        f = interpolate.interp1d(self.fp_grid, self.fp_potential)
        fp_pot = f(self.z_grid_points)
        return fp_pot - self.potential
