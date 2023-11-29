# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from monty.serialization import loadfn

from pydefect_2d.correction.gauss_energy import GaussEnergy
from pydefect_2d.one_d.slab_model import OneDSlabModel


def test_one_d_slab_model(test_files):
    diele_dist = loadfn(test_files / "dielectric_const_dist.json")
    charge = loadfn(test_files / "1d_gauss" / "1d_gauss_charge_0.500.json")
    potential = loadfn(test_files / "1d_gauss" / "1d_gauss_potential_0.500.json")
    fp_pot = loadfn(test_files / "H_ad_1" / "1d_fp_potential.json")
    gauss_energy = GaussEnergy(z=0.5,
                               isolated_energy=10.12345,
                               periodic_energy=0.12348)

    one_d_slab_model = OneDSlabModel(charge_state=1,
                                   diele_dist=diele_dist,
                                   one_d_gauss_charge=charge,
                                   one_d_gauss_potential=potential,
                                   one_d_fp_potential=fp_pot,
                                   gauss_energy=gauss_energy)
    print(one_d_slab_model)