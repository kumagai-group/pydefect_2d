# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import pytest
from monty.serialization import loadfn

from pydefect_2d.correction.isolated_gauss import IsolatedGaussEnergy, \
    CalcIsolatedGaussEnergy


@pytest.fixture
def gauss_energy(test_files):
    gauss = loadfn(test_files / "correction" / "gauss_charge_model_0.370.json")
    diele = loadfn(test_files / "dielectric_const_dist.json")

    calc = CalcIsolatedGaussEnergy(gauss_charge_model=gauss,
                                   diele_const_dist=diele,
                                   k_max=1,
                                   k_mesh_dist=0.05,
                                   multiprocess=True)
    return calc.isolated_gauss_energy


def test_isolated_gaussian_energy(gauss_energy: IsolatedGaussEnergy):
    assert gauss_energy.self_energy == 1.0186539512068704


