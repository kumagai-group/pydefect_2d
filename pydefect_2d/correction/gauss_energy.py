# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from matplotlib.axes import Axes
from monty.json import MSONable
from monty.serialization import loadfn
from scipy.interpolate import interpolate
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.util.utils import add_z_to_filename, get_z_from_filename
from pydefect_2d.correction.isolated_gauss import IsolatedGaussEnergy
from pydefect_2d.three_d.slab_model import GaussChargePotential, \
    GaussChargeModel, electrostatic_energy_at_q1


@dataclass
class GaussEnergy:
    z: float
    isolated_energy: float
    periodic_energy: float


@dataclass
class GaussEnergies(MSONable, ToJsonFileMixIn):
    gauss_energies: List[GaussEnergy]

    def to_plot(self, ax: Axes, charge=1):
        zs, corr = [], []
        for ge in self.gauss_energies:
            zs.append(ge.z)
            corr.append((ge.isolated_energy - ge.periodic_energy) * charge)

        ax.set_xlabel("Position in frac. coord.")
        ax.set_ylabel("Correction energy (eV)")
        ax.scatter(zs, corr, label="corr", color="black")

        if len(zs) >= 3:
            ip_ = interpolate.interp1d(zs, corr, kind="cubic")
            ip_xs = np.linspace(min(zs), max(zs), 100)
            ip_ys = ip_(ip_xs)
            ax.plot(ip_xs, ip_ys)

        ax.set_ylim(min(corr) - 0.2, max(corr) + 0.2)


def make_gauss_energy(corr_dir: Path, z: float):
    charge_fname = add_z_to_filename("gauss_charge_model.json", z)
    pot_fname = add_z_to_filename("gauss_charge_potential.json", z)
    iso_fname = add_z_to_filename("isolated_gauss_energy.json", z)

    charge: GaussChargeModel = loadfn(corr_dir / charge_fname)
    pot: GaussChargePotential = loadfn(corr_dir / pot_fname)
    isolated: IsolatedGaussEnergy = loadfn(corr_dir / iso_fname)
    periodic_energy = electrostatic_energy_at_q1(pot, charge)
    return GaussEnergy(z, isolated.self_energy, periodic_energy)


def make_gauss_energies(corr_dir: Path,
                        z_range: List[float] = None):
    zs = []
    for f in glob.glob(str(corr_dir / "isolated_gauss_energy*.json")):
        z = get_z_from_filename(f)
        if z_range is None or z_range[0] <= z <= z_range[1]:
            zs.append(z)

    result = [make_gauss_energy(corr_dir, z) for z in zs]
    return GaussEnergies(result)
