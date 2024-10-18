# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from matplotlib.axes import Axes
from monty.json import MSONable
from monty.serialization import loadfn
from pydefect.cli.main_tools import parse_dirs
from scipy.interpolate import interpolate
from scipy.optimize import fsolve
from tabulate import tabulate
from vise.util.logger import get_logger
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.correction.isolated_gauss import IsolatedGaussEnergy
from pydefect_2d.dielectric.make_extended_diele_dist import ChangeVacuum
from pydefect_2d.three_d.grids import Grid
from pydefect_2d.three_d.slab_model import GaussChargeModel, \
    CalcGaussChargePotential, electrostatic_energy_at_q1, SlabModel

logger = get_logger(__name__)


@dataclass
class SpecialVacuum(MSONable, ToJsonFileMixIn):
    lengths: List[float]
    electrostatic_energies: List[float]
    isolated_gauss_energy: IsolatedGaussEnergy

    @property
    def iso_energy(self) -> float:
        return self.isolated_gauss_energy.self_energy

    @property
    def relative_energies(self) -> List[float]:
        return [e - self.iso_energy for e in self.electrostatic_energies]

    def __str__(self):
        data = [[length, energy] for length, energy
                in zip(self.lengths, self.electrostatic_energies)]
        iso = str(self.isolated_gauss_energy)
        table = tabulate(data,
                         headers=["Length (Å)", "Electrostatic energy (eV)"],
                         floatfmt=".2f", tablefmt="simple")
        try:
            sv_length = fsolve(self.rel_interpolate, np.array([0.0]))
            length_str = f"Special vacuum length: {sv_length[0]:.3f}"
        except:
            length_str = ("Special vacuum length is not determined. "
                          "Check the lengths and electrostatic energies")

        return "\n".join([iso, table, length_str])

    @property
    def rel_interpolate(self):
        return interpolate.interp1d(self.lengths, self.relative_energies,
                                    fill_value="extrapolate")

    @property
    def interpolate(self):
        return interpolate.interp1d(self.lengths, self.electrostatic_energies,
                                    fill_value="extrapolate")

    def to_plot(self, ax: Axes):
        ax.set_xlabel("Vertical distance (Å)")
        ax.set_ylabel("Electrostatic energy (V)")
        ax.scatter(self.lengths, self.electrostatic_energies)

        ax.axhline(self.iso_energy, color="black", linestyle="--")

        xs = np.linspace(min(self.lengths), max(self.lengths), num=100)
        ys = [self.interpolate(x) for x in xs]
        ax.plot(xs, ys, '-')


def extend_dielectric_const_dist(args):
    for new_l in args.slab_lengths:
        logger.info(f"Extending dielectric constant distribution to {new_l}.")
        p = Path(f"{new_l}A")
        p.mkdir(exist_ok=True)

        change_vac = ChangeVacuum(args.diele_dist, new_l=new_l)
        diele = change_vac.diele_const_dist
        diele.to_json_file(p / "dielectric_const_dist.json")

        new_g_chg_model: GaussChargeModel = deepcopy(args.gauss_charge_model)
        orig_z_grid = new_g_chg_model.grids.z_grid
        new_num_grid = int(orig_z_grid.num_grid * new_l / orig_z_grid.length / 2) * 2
        z_grid = Grid(new_l, new_num_grid)

        new_g_chg_model.grids.z_grid = z_grid
        new_g_chg_model.set_periodic_gauss_charge_profile()
        new_g_chg_model.to_json_file(p / "gauss_charge_model.json")

        calc_pot = CalcGaussChargePotential(diele, new_g_chg_model)
        calc_pot.potential.to_json_file(p / "gauss_charge_potential.json")


def calc_special_vacuum(args):

    lengths, energies = [], []

    def _inner(_dir: Path):
        slab_model = SlabModel(loadfn(_dir / "dielectric_const_dist.json"),
                               loadfn(_dir / "gauss_charge_model.json"),
                               loadfn(_dir / "gauss_charge_potential.json"),
                               charge_state=1)

        energies.append(slab_model.electrostatic_energy)
        lengths.append(slab_model.z_length)

    parse_dirs(args.dirs, _inner, True)
    sv = SpecialVacuum(lengths, energies, args.isolated_gauss_energy)
    sv.to_json_file()
    print(sv)
