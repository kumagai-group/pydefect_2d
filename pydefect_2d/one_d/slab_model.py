# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass

from monty.json import MSONable
from tabulate import tabulate
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.correction.gauss_energy import GaussEnergy
from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.one_d.charge import OneDGaussChargeModel
from pydefect_2d.one_d.potential import OneDGaussPotential, OneDFpPotential
from pydefect_2d.three_d.grids import Grid


@dataclass
class OneDSlabModel(MSONable, ToJsonFileMixIn):
    charge: int
    diele_dist: DielectricConstDist  # [ε_x, ε_y, ε_z] as a function of z
    one_d_gauss_charge: OneDGaussChargeModel  # q is already multiplied.
    one_d_gauss_potential: OneDGaussPotential  # q is already multiplied.
    one_d_fp_potential: OneDFpPotential
    gauss_energy: GaussEnergy

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

    def __str__(self):
        header = ["pos (Å)", "charge", "potential"]
        list_ = [[z, charge, pot] for z, charge, pot in
                 zip(self.grid.grid_points(),
                     self.one_d_gauss_charge.periodic_charges,
                     self.one_d_gauss_potential.potential)]
        result = [tabulate(list_, tablefmt="plain", headers=header)]

        volume = self.grid.length * self.one_d_gauss_charge.surface
        integrated = self.one_d_gauss_charge.periodic_charges.mean() * volume
        d = [[f"Charge (|e|)", self.charge],
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
