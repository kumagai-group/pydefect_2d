# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass

from monty.json import MSONable
from tabulate import tabulate
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.one_d.charge import OneDGaussChargeModel
from pydefect_2d.one_d.potential import OneDGaussPotential, OneDFpPotential
from pydefect_2d.three_d.grids import Grid


@dataclass
class OneDSlabModel(MSONable, ToJsonFileMixIn):
    charge: int
    diele_dist: DielectricConstDist  # [ε_x, ε_y, ε_z] as a function of z
    gauss_1d_charge: OneDGaussChargeModel  # q is already multiplied.
    gauss_1d_potential: OneDGaussPotential  # q is already multiplied.
    fp_1d_potential: OneDFpPotential
    isolated_energy: float
    periodic_energy: float

    def __post_init__(self):
        assert (self.diele_dist.dist.length
                == self.gauss_1d_potential.grid.length)

    @property
    def correction_energy(self):
        return self.isolated_energy - self.periodic_energy

    @property
    def grid(self) -> Grid:
        return self.gauss_1d_potential.grid

    def __str__(self):
        header = ["pos (Å)", "charge", "potential"]
        list_ = [[z, charge, pot] for z, charge, pot in
                 zip(self.grid.grid_points(),
                     self.gauss_1d_charge.periodic_charges,
                     self.gauss_1d_potential.potential)]
        result = [tabulate(list_, tablefmt="plain", headers=header)]

        volume = self.grid.length * self.gauss_1d_charge.surface
        integrated = self.gauss_1d_charge.periodic_charges.mean() * volume
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
        grid_idx, z = self.gauss_1d_charge.farthest_z_from_defect
        gauss_pot = self.gauss_1d_potential.potential[grid_idx]
        fp_pot = self.fp_1d_potential.potential_func(z)
        return fp_pot - gauss_pot
