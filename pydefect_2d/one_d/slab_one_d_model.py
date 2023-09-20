# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from functools import cached_property

from monty.json import MSONable
from tabulate import tabulate
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.one_d.one_d_charge import OneDGaussChargeModel
from pydefect_2d.one_d.one_d_potential import Gauss1DPotential, Fp1DPotential
from pydefect_2d.potential.grids import Grid
from pydefect_2d.potential.slab_model import electrostatic_energy_at_q1


@dataclass
class Slab1DModel(MSONable, ToJsonFileMixIn):
    diele_dist: DielectricConstDist  # [ε_x, ε_y, ε_z] as a function of z
    gauss_1d_charge: OneDGaussChargeModel  # q is already multiplied.
    gauss_1d_potential: Gauss1DPotential  # q is already multiplied.
    fp_1d_potential: Fp1DPotential
    correction_energy: float

    def __post_init__(self):
        assert (self.diele_dist.dist.length
                == self.gauss_1d_potential.grid.length)

    @property
    def grid(self) -> Grid:
        return self.gauss_1d_potential.grid

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
        fp_pot = self.fp_potential.potential_func(z)
        return fp_pot - gauss_pot
