# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass

import numpy as np
from monty.json import MSONable
from tabulate import tabulate
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.correction.gauss_energy import GaussEnergy
from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.one_d.charge import OneDGaussChargeModel
from pydefect_2d.one_d.potential import OneDGaussPotential, OneDFpPotential
from pydefect_2d.three_d.grids import Grid
from pydefect_2d.three_d.slab_model_plotter import SlabModelPlotAbs
from pydefect_2d.util.utils import with_end_point


@dataclass
class OneDSlabModel(MSONable, ToJsonFileMixIn, SlabModelPlotAbs):
    charge_state: int
    diele_dist: DielectricConstDist  # [ε_x, ε_y, ε_z] as a function of z
    one_d_gauss_charge: OneDGaussChargeModel  # q is already multiplied.
    one_d_gauss_potential: OneDGaussPotential  # q is already multiplied.
    one_d_fp_potential: OneDFpPotential
    gauss_energy: GaussEnergy # q is already multiplied.

    @property
    def isolated_energy(self):
        return self.gauss_energy.isolated_energy

    @property
    def periodic_energy(self):
        return self.gauss_energy.periodic_energy

    def __post_init__(self):
        assert (self.diele_dist.dist.length
                == self.one_d_gauss_potential.grid.length)

    @property
    def correction_energy(self):
        return self.isolated_energy - self.periodic_energy

    @property
    def grid(self) -> Grid:
        return self.one_d_gauss_potential.grid

    def get_xy_ave_potential(self, frac_coord):
        z_num_grid = self.grid.num_grid
        z_frac_coords = np.linspace(0, 1, z_num_grid + 1)
        idx = (np.abs(z_frac_coords - frac_coord)).argmin()
        return float(self.one_d_gauss_potential.potential[idx])

    def __str__(self):
        header = ["pos (Å)", "charge", "potential"]
        list_ = [[z, charge, pot] for z, charge, pot in
                 zip(self.grid.grid_points(),
                     self.one_d_gauss_charge.periodic_charges,
                     self.one_d_gauss_potential.potential)]
        result = [tabulate(list_, tablefmt="plain", headers=header)]

        volume = self.grid.length * self.one_d_gauss_charge.surface
        integrated = self.one_d_gauss_charge.periodic_charges.mean() * volume
        d = [[f"Charge (|e|)", self.charge_state],
             [f"Integrated charge (|e|)", integrated],
             [f"Isolated energy (eV)", self.isolated_energy],
             [f"Periodic energy (eV)", self.periodic_energy],
             [f"Correction energy w/o alignment (eV)", self.correction_energy],
             [f"Potential diff", self.correction_energy]]
        result.append(tabulate(d, tablefmt='simple', floatfmt=".4"))
        return "\n".join(result)

    @property
    def potential_diff(self):
        grid_idx, z = self.one_d_gauss_charge.farthest_z_from_defect
        gauss_pot = self.one_d_gauss_potential.potential[grid_idx]
        fp_pot = self.one_d_fp_potential.potential_func(z)
        return fp_pot - gauss_pot

    def gauss_charge_z_plot(self, ax):
        self.one_d_gauss_charge.to_plot(ax)

    def gauss_potential_z_plot(self, ax):
        self.one_d_gauss_potential.to_plot(ax, label="Gauss potential")

    def fp_potential_plot(self, ax):
        ax.plot(self.one_d_fp_potential.grid.grid_points(True),
                with_end_point(self.one_d_fp_potential.potential),
                label="FP", color="blue")

        grid_pts = self.one_d_gauss_potential.grid.grid_points(True)
        fp_pot = self.one_d_fp_potential.potential_func(grid_pts)
        diff_pot = fp_pot - with_end_point(self.one_d_gauss_potential.potential)
        ax.plot(grid_pts, diff_pot, label="diff", color="green", linestyle=":")
        ax.legend()

    def epsilon_plot(self, ax):
        ax.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
        z_grid_pts = self.diele_dist.dist.grid_points(True)
        for e, direction in zip(self.diele_dist.static, ["x", "y", "z"]):
            ax.plot(z_grid_pts, with_end_point(e), label=direction)
        ax.legend()

    @property
    def z_length(self):
        return self.diele_dist.dist.length
