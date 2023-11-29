# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import multiprocessing as multi
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from math import pi
from multiprocessing import Pool
from typing import Tuple

import numpy as np
from monty.json import MSONable
from numpy import exp
from scipy.constants import epsilon_0, elementary_charge, angstrom
from scipy.fftpack import ifftn, fftn
from scipy.linalg import solve
from tabulate import tabulate
from tqdm import tqdm
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.three_d.grids import Grids, Grid
from pydefect_2d.one_d.potential import OneDFpPotential
from pydefect_2d.three_d.slab_model_plotter import SlabModelPlotAbs
from pydefect_2d.util.utils import with_end_point


@dataclass
class GaussChargeModel(MSONable, ToJsonFileMixIn):
    """Gauss charge model with 1|e| under periodic boundary condition. """
    grids: Grids
    std_dev: float
    gauss_pos_in_frac: float  # in fractional coord. x=y=0
    periodic_charges: np.array = None

    def __str__(self):
        charge = self.periodic_charges.mean() * self.grids.volume
        result = [f"Standard deviation (A): {self.std_dev:.2}",
                  f"Charge sum (|e|): {charge:.3}"]
        # return "\n".join(result)

    def __post_init__(self):
        if self.periodic_charges is None:
            self.periodic_charges = self._make_periodic_gauss_charge_profile

    @property
    def gauss_pos_in_cart(self):
        return self.gauss_pos_in_frac * self.grids.z_grid.length

    @property
    def _make_periodic_gauss_charge_profile(self):
        coefficient = 1 / self.std_dev ** 3 / (2 * pi) ** 1.5

        (nx, ny), nz = self.grids.xy_grids.num_grids, self.grids.z_grid.num_grid
        gauss = np.zeros([nx, ny, nz])

        xy2 = self.grids.xy_grids.squared_length_on_grids()
        for nz, lz in enumerate(self.grids.z_grid.grid_points()):
            xyz2 = xy2 + self._min_z2(lz)
            gauss[:, :, nz] = exp(- xyz2 / (2 * self.std_dev ** 2))

        return coefficient * gauss

    def _min_z2(self, lz):
        return min(
            [abs(lz - self.grids.z_grid.length * (i + self.gauss_pos_in_frac))
             for i in range(-1, 2)]
        ) ** 2

    @cached_property
    def reciprocal_charge(self):
        result = fftn(self.periodic_charges)
        result[0, 0, 0] = 0  # introduce background charge
        return result

    @cached_property
    def xy_average_charge(self) -> np.array:
        return np.real(self.periodic_charges.mean(axis=(0, 1)))

    @cached_property
    def xy_integrated_charge(self) -> np.array:
        return self.xy_average_charge * self.grids.xy_grids.xy_area

    @property
    def farthest_z_from_defect(self) -> Tuple[int, float]:
        rel_z_in_frac = (self.gauss_pos_in_frac + 0.5) % 1.
        z = self.grids.z_grid.length * rel_z_in_frac
        return self.grids.z_grid.nearest_grid_point(z)

    def to_plot(self, ax, charge=1):
        ax.set_ylabel("Charge (|e|/Å)")
        ys = with_end_point(self.xy_integrated_charge * charge)
        ax.plot(self.grids.z_grid.grid_points(True),
                ys, label="charge", color="black")


@dataclass
class GaussChargePotential(MSONable, ToJsonFileMixIn):
    grids: Grids  # assume orthogonal system
    potential: np.ndarray  # potential for positive charge

    @cached_property
    def xy_ave_potential(self):
        return np.real(self.potential.mean(axis=(0, 1)))


    def get_potential(self, cart_coord):
        x, y, z = cart_coord
        xi, yi = self.grids.xy_grids.nearest_grid_point(x, y)[0]
        zi = self.grids.z_grid.nearest_grid_point(z)[0]
        return self.potential[xi, yi, zi]

    def to_plot(self, ax, charge=1):
        ax.set_ylabel("Potential energy (eV)")
        ys = with_end_point(self.potential.mean(axis=(0, 1)) * charge)
        ax.plot(self.grids.z_grid.grid_points(True),
                ys, label="Gauss model potential", color="red")
        ax.legend()


@dataclass
class CalcGaussChargePotential:
    dielectric_const: DielectricConstDist  # [ε_x, ε_y, ε_z] as a function of z
    gauss_charge_model: GaussChargeModel  # assume orthogonal system
    multiprocess: bool = True

    def __post_init__(self):
        try:
            assert (self.dielectric_const.dist.length
                    == self.gauss_charge_model.grids.z_grid.length)
        except AssertionError:
            e_z_dist = self.dielectric_const.dist
            g_z_grid = self.gauss_charge_model.grids.z_grid

            print(f"epsilon z lattice length {e_z_dist.length}")
            print(f"epsilon num grid {e_z_dist.num_grid}")
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
        x_rec_e, y_rec_e, z_rec_e = self.dielectric_const.reciprocal_static

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

        inv_pot_by_mz = solve(factors, rec_chg * z_num_grid, assume_a="her")
        return i_ga, i_gb, inv_pot_by_mz

    @cached_property
    def reciprocal_potential(self):
        x_grids, y_grids = self.xy_num_grids

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
class SlabModel(MSONable, ToJsonFileMixIn, SlabModelPlotAbs):
    diele_dist: DielectricConstDist  # [ε_x, ε_y, ε_z] as a function of z
    gauss_charge_model: GaussChargeModel
    gauss_charge_potential: GaussChargePotential
    charge_state: int
    fp_potential: OneDFpPotential = None

    def __post_init__(self):
        assert (self.diele_dist.dist.length
                == self.gauss_charge_model.grids.z_grid.length)

    @property
    def grids(self) -> Grids:
        return self.gauss_charge_model.grids

    def gauss_charge_z_plot(self, ax):
        self.gauss_charge_model.to_plot(ax, charge=self.charge_state)

    def gauss_potential_z_plot(self, ax):
        self.gauss_charge_potential.to_plot(ax, charge=self.charge_state)

    def fp_potential_plot(self, ax):
        ax.plot(self.fp_potential.grid.grid_points(True),
                with_end_point(self.fp_potential.potential),
                label="FP", color="blue")

        z_grid_pts = self.gauss_charge_potential.grids.z_grid.grid_points(True)
        fp_pot = self.fp_potential.potential_func(z_grid_pts)
        diff_pot = fp_pot - with_end_point(self.xy_ave_pot)
        ax.plot(z_grid_pts, diff_pot,
                label="diff", color="green", linestyle=":")

    def epsilon_plot(self, ax):
        ax.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
        z_grid_pts = self.diele_dist.dist.grid_points(True)
        for e, direction in zip(self.diele_dist.static, ["x", "y", "z"]):
            ax.plot(z_grid_pts, with_end_point(e), label=direction)
        ax.legend()

    @property
    def z_length(self):
        return self.diele_dist.dist.length

    @cached_property
    def electrostatic_energy(self) -> float:
        x = electrostatic_energy_at_q1(self.gauss_charge_potential,
                                       self.gauss_charge_model)
        return x * self.charge_state ** 2

    @cached_property
    def xy_integrated_charge(self):
        return self.gauss_charge_model.xy_integrated_charge * self.charge_state

    @cached_property
    def xy_ave_pot(self):
        return self.gauss_charge_potential.xy_ave_potential * self.charge_state

    def __str__(self):
        header = ["pos (Å)", "charge", "potential"]
        list_ = [[z, charge, pot] for z, charge, pot in
                 zip(self.grids.z_grid.grid_points(),
                     self.xy_integrated_charge,
                     self.xy_ave_pot)]
        result = [tabulate(list_, tablefmt="plain", headers=header)]

        integrated_charge = (self.gauss_charge_model.periodic_charges.mean()
                             * self.grids.volume * self.charge_state)
        result.append(f"Integrated charge (|e|): {integrated_charge:.3}")
        result.append(f"Electrostatic energy (eV): "
                      f"{self.electrostatic_energy:.3}")
        return "\n".join(result)

    @property
    def potential_diff(self):
        grid_idx, z = self.gauss_charge_model.farthest_z_from_defect
        gauss_pot = self.xy_ave_pot[grid_idx]
        if self.fp_potential:
            fp_pot = self.fp_potential.potential_func(z)
            return fp_pot - gauss_pot
        else:
            return None


def electrostatic_energy_at_q1(potential: GaussChargePotential,
                               chg_model: GaussChargeModel) -> float:
    pot = potential.potential
    chg = chg_model.periodic_charges
    vol = chg_model.grids.volume
    return np.real((np.mean(pot * chg) * vol / 2))


