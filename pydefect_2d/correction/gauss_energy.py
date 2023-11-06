# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import glob
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List

import numpy as np
from matplotlib.axes import Axes
from monty.json import MSONable
from monty.serialization import loadfn
from scipy.interpolate import interpolate
from vise.util.logger import get_logger
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.util.utils import add_z_to_filename, get_z_from_filename
from pydefect_2d.correction.isolated_gauss import IsolatedGaussEnergy
from pydefect_2d.three_d.slab_model import GaussChargePotential, \
    GaussChargeModel, electrostatic_energy_at_q1


logger = get_logger(__name__)


@dataclass
class GaussEnergy:
    z: float
    isolated_energy: float
    periodic_energy: float


@dataclass
class GaussEnergies(MSONable, ToJsonFileMixIn):
    gauss_energies: List[GaussEnergy]
    ip_kind: str = "cubic"
    num_ip: int = 100

    @cached_property
    def zs(self):
        return [ge.z for ge in self.gauss_energies]

    @property
    def isolated(self):
        return [ge.isolated_energy for ge in self.gauss_energies]

    @property
    def periodic(self):
        return [ge.periodic_energy for ge in self.gauss_energies]

    @property
    def corrections(self):
        return [i - p for i, p in zip(self.isolated, self.periodic)]

    @property
    def ip_zs(self):
        return np.linspace(min(self.zs), max(self.zs), self.num_ip)

    def ip_(self, type_: str):
        pts = getattr(self, type_)
        try:
            return interpolate.interp1d(self.zs, pts, kind=self.ip_kind)
        except ValueError:
            logger.info("More than 3 data points are needed.")
            raise

    def to_plot(self, ax: Axes):
        for type_, color in [["periodic", "red"],
                             ["isolated", "blue"],
                             ["corrections", "black"]]:
            pts = getattr(self, type_)
            ax.scatter(self.zs, pts, label=type_, color=color)
            if len(self.zs) > 3:
                ax.plot(self.ip_zs, self.ip_(type_)(self.ip_zs), color=color)

        ax.set_xlabel("Position in frac. coord.")
        ax.set_ylabel("Energy (eV)")
        ax.legend()

    def get_gauss_energy(self, z):
        p, i = float(self.ip_("periodic")(z)), float(self.ip_("isolated")(z))
        return GaussEnergy(z=z, isolated_energy=i, periodic_energy=p)


def make_gauss_energy(corr_dir: Path, z: float):
    charge_fname = add_z_to_filename("gauss_charge_model.json", z)
    pot_fname = add_z_to_filename("gauss_charge_potential.json", z)
    iso_fname = add_z_to_filename("isolated_gauss_energy.json", z)

    charge: GaussChargeModel = loadfn(corr_dir / charge_fname)
    pot: GaussChargePotential = loadfn(corr_dir / pot_fname)
    isolated: IsolatedGaussEnergy = loadfn(corr_dir / iso_fname)
    periodic_energy = electrostatic_energy_at_q1(pot, charge)
    return GaussEnergy(z, isolated.self_energy, periodic_energy)


def make_gauss_energies(corr_dir: Path, z_range: List[float] = None):
    zs = []
    for f in glob.glob(str(corr_dir / "isolated_gauss_energy*.json")):
        z = get_z_from_filename(f)
        if z_range is None or z_range[0] <= z <= z_range[1]:
            zs.append(z)

    result = [make_gauss_energy(corr_dir, z) for z in zs]
    return GaussEnergies(result)
