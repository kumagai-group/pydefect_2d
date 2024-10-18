# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from argparse import Namespace

from matplotlib import pyplot as plt
from monty.serialization import loadfn

from pydefect_2d.cli.special_vacuum.main_function import \
    extend_dielectric_const_dist, SpecialVacuum


def test_special_vacuum(mocker):
    iso = mocker.stub()
    iso.self_energy = 1.0

    sv = SpecialVacuum(lengths=[10, 20, 30],
                       electrostatic_energies=[-0.5, 1.5, 2.5],
                       isolated_gauss_energy=iso)
    sv.to_plot(plt.gca())
    print(sv)
    plt.show()


def test_extend_dielectric_const_dist(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    diele = loadfn(test_files / "dielectric_const_dist.json")
    gauss_charge = loadfn(test_files / "correction" / "gauss_charge_model_0.500.json")
    args = Namespace(slab_lengths=[20.0],
                     diele_dist=diele,
                     gauss_charge_model=gauss_charge)
    extend_dielectric_const_dist(args)
