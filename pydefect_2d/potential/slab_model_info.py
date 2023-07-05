# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import multiprocessing as multi
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from math import pi
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
from monty.json import MSONable
from numpy import exp
from scipy.constants import epsilon_0, elementary_charge, angstrom
from scipy.fftpack import ifftn, fftn
from scipy.interpolate import interp1d
from tabulate import tabulate
from tqdm import tqdm
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.potential.epsilon_distribution import EpsilonDistribution
from pydefect_2d.potential.grids import Grid, Grids


@dataclass
class GaussChargeModel(MSONable, ToJsonFileMixIn):
    """Gauss charge model with 1|e| under periodic boundary condition. """
    grids: Grids
    sigma: float
    defect_z_pos_in_frac: float  # in fractional coord. x=y=0
    epsilon_x: np.array
    epsilon_y: np.array
    charges: np.array = None

    def __str__(self):
        charge = self.charges.mean() * self.grids.volume
        result = [f"sigma (A): {self.sigma:.2}",
                  f"Charge sum (|e|): {charge:.3}"]
        return "\n".join(result)

    def __post_init__(self):
        assert len(self.epsilon_x) == len(self.epsilon_y) == self.grids.z_grid.num_grid

        if self.charges is None:
            self.charges = self._make_gauss_charge_profile

    @property
    def defect_z_pos_in_length(self):
        return self.defect_z_pos_in_frac * self.grids.z_length

    @property
    def epsilon_ave(self):
        return np.sqrt(self.epsilon_x * self.epsilon_y)

    @property
    def square_x_scaling(self):
        return self.epsilon_ave / self.epsilon_x

    @property
    def square_y_scaling(self):
        return self.epsilon_ave / self.epsilon_y

    @property
    def _make_gauss_charge_profile(self):
        coefficient = 1 / self.sigma ** 3 / (2 * pi) ** 1.5

        (nx, ny), nz = self.grids.xy_grids.num_grids, self.grids.z_grid.num_grid
        gauss = np.zeros([nx, ny, nz])

        xy2 = self.grids.xy_grids.squared_length_on_grids
        for nz, lz in enumerate(self.grids.z_grid_points):
            gauss[:, :, nz] = exp(-(xy2 + self._z2(lz)) / (2 * self.sigma ** 2))

        return coefficient * gauss

    def _z2(self, lz):
        return min(
            [abs(lz - self.grids.z_length * (i + self.defect_z_pos_in_frac))
             for i in range(-1, 2)]
        ) ** 2

    @cached_property
    def reciprocal_charge(self):
        result = fftn(self.charges)
        result[0, 0, 0] = 0  # introduce background charge
        return result

    @cached_property
    def xy_integrated_charge(self) -> np.array:
        xy_average = np.real(self.charges.mean(axis=(0, 1)))
        return xy_average * self.grids.xy_grids.area

    @property
    def farthest_z_from_defect(self) -> Tuple[int, float]:
        rel_z_in_frac = (self.defect_z_pos_in_frac + 0.5) % 1.
        z = self.grids.z_length * rel_z_in_frac
        return self.grids.nearest_z_grid_point(z)

    def to_plot(self, ax, charge=1):
        ax.set_ylabel("Charge (|e|/Å)")
        ax.plot(self.grids.z_grid_points, self.xy_integrated_charge * charge,
                label="charge", color="black")


@dataclass
class GaussChargePotential(MSONable, ToJsonFileMixIn):
    grids: Grids  # assume orthogonal system
    potential: np.array  # potential for positive charge

    @cached_property
    def xy_ave_potential(self):
        return np.real(self.potential.mean(axis=(0, 1)))

    def to_plot(self, ax, charge=1):
        ax.set_ylabel("Potential energy (eV)")
        ax.plot(self.grids.z_grid_points,
                self.potential.mean(axis=(0, 1)) * charge,
                label="Gauss model potential", color="red")
        ax.legend()


@dataclass
class CalcGaussChargePotential:
    epsilon: EpsilonDistribution  # [epsilon_x, epsilon_y, epsilon_z] along z
    gauss_charge_model: GaussChargeModel  # assume orthogonal system
    multiprocess: bool = True

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
    def xy_grids(self):
        return self.gauss_charge_model.grids.xy_grids

    @property
    def z_grid(self):
        return self.gauss_charge_model.grids.z_grid

    @property
    def xy_num_grids(self):
        return self.xy_grids.num_grids

    @property
    def num_grids(self):
        return self.xy_grids.num_grids + [self.z_grid.num_grid]

    def _solve_poisson_eq(self, ab_grid_idx):
        i_ga, i_gb = ab_grid_idx

        z_num_grid = self.gauss_charge_model.grids.z_grid.num_grid
        x_rec_e, y_rec_e, z_rec_e = self.epsilon.reciprocal_static
        rec_chg = self.gauss_charge_model.reciprocal_charge[i_ga, i_gb, :]

        factors = []
        Gzs = self.gauss_charge_model.grids.z_grid.Gs
        for i_gz, gz in enumerate(Gzs):
            inv_rho_by_mz = [x_rec_e[i_gz - i_gz_prime] * self.Ga2s[i_ga] +
                             y_rec_e[i_gz - i_gz_prime] * self.Gb2s[i_gb] +
                             z_rec_e[i_gz - i_gz_prime] * gz * gz_prime
                             for i_gz_prime, gz_prime in enumerate(Gzs)]
            if i_ga == 0 and i_gb == 0 and i_gz == 0:
                # To avoid a singular error, any non-zero value needs to be set.
                inv_rho_by_mz[0] = 1.0
            factors.append(inv_rho_by_mz)
        factors = np.array(factors)
        inv_pot_by_mz = np.linalg.solve(factors, rec_chg * z_num_grid)
        return i_ga, i_gb, inv_pot_by_mz

    @cached_property
    def reciprocal_potential(self):
        x_grids, y_grids = self.num_grids[:2]

        result = np.zeros(self.num_grids, dtype=np.complex_)
        grids = [[i_gx, i_gy] for i_gx, i_gy
                 in product(range(x_grids), range(y_grids))]

        if self.multiprocess:
            p = Pool(multi.cpu_count())
            with p:
                collected_data = p.map(self._solve_poisson_eq, tqdm(grids))
        else:
            collected_data = [self._solve_poisson_eq(g) for g in grids]

        for d in collected_data:
            result[d[0], d[1], :] = d[2]

        result[0, 0, 0] = 0
        return result / epsilon_0 * elementary_charge / angstrom

    @cached_property
    def potential(self):
        real = ifftn(self.reciprocal_potential)
        return GaussChargePotential(self.gauss_charge_model.grids, real)


@dataclass
class FP1dPotential(MSONable, ToJsonFileMixIn):
    grid: Grid
    potential: List[float]

    @cached_property
    def interpol_pot_func(self):
        return interp1d(self.grid.grid_points, self.potential)

    def to_plot(self, ax):
        ax.set_ylabel("Potential (V)")
        ax.plot(self.grid.grid_points, self.potential,
                label="potential", color="blue")


@dataclass
class SlabModel(MSONable, ToJsonFileMixIn):
    epsilon: EpsilonDistribution  # [epsilon_x, epsilon_y, epsilon_z] along z
    gauss_charge_model: GaussChargeModel
    gauss_charge_potential: GaussChargePotential
    charge: int
    fp_potential: FP1dPotential = None

    def __post_init__(self):
        assert self.epsilon.grid == self.gauss_charge_model.grids.z_grid
#        assert self.gauss_charge_model.grids == self.gauss_charge_potential.grids

    @property
    def grids(self) -> Grids:
        return self.gauss_charge_model.grids

    @cached_property
    def electrostatic_energy(self) -> float:
        result_at_charge1 = np.real(
            (np.mean(self.gauss_charge_potential.potential * self.gauss_charge_model.charges)
             * self.gauss_charge_model.grids.volume / 2))
        return result_at_charge1 * self.charge ** 2

    @cached_property
    def xy_charge(self):
        return self.gauss_charge_model.xy_integrated_charge * self.charge

    @cached_property
    def xy_potential(self):
        return self.gauss_charge_potential.xy_ave_potential * self.charge

    def __str__(self):
        header = ["pos (Å)", "charge", "potential"]
        list_ = [[pos, charge, pot] for pos, charge, pot in
                 zip(self.grids.z_grid_points, self.xy_charge, self.xy_potential)]
        result = [tabulate(list_, tablefmt="plain", headers=header)]

        integrated_charge = (self.gauss_charge_model.charges.mean()
                             * self.grids.volume * self.charge)
        result.append(f"Integrated charge (|e|): {integrated_charge:.3}")
        result.append(f"Electrostatic energy (eV): "
                      f"{self.electrostatic_energy:.3}")
        return "\n".join(result)

    @property
    def potential_diff(self):
        if self.fp_potential is None:
            return
        grid_idx, z = self.gauss_charge_model.farthest_z_from_defect
        gauss_pot = self.xy_potential[grid_idx]
        fp_pot = self.fp_potential.interpol_pot_func(z)
        return fp_pot - gauss_pot

