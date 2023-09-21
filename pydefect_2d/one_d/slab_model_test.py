# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from monty.serialization import loadfn

from pydefect_2d.one_d.slab_model import OneDSlabModel


def test_one_d_slab_model(test_files):
    diele_dist = loadfn(test_files / "dielectric_const_dist.json")
    charge = loadfn(test_files / "1d_gauss" / "gauss_1d_charge_0.500.json")
    potential = loadfn(test_files / "1d_gauss" / "gauss_1d_potential_0.500.json")
    fp_pot = loadfn(test_files / "H_ad_1" / "fp_1d_potential.json")

    one_d_pot_diff = OneDSlabModel(charge=1,
                                   diele_dist=diele_dist,
                                   gauss_1d_charge=charge,
                                   gauss_1d_potential=potential,
                                   fp_1d_potential=fp_pot,
                                   isolated_energy=10.12345,
                                   periodic_energy=0.12348)

    print(one_d_pot_diff)
