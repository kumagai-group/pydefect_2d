# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest
from matplotlib import pyplot as plt
from vise.tests.helpers.assertion import assert_json_roundtrip

from pydefect_2d.potential.distribution import Dist, ManualDist, GaussianDist
from pydefect_2d.potential.epsilon_distribution import \
    DielectricConstDist
from pydefect_2d.potential.grids import Grid, Grids, XYGrids
from pydefect_2d.potential.plotter import ProfilePlotter
from pydefect_2d.potential.slab_model_info import CalcGaussChargePotential, \
    GaussChargeModel, SlabModel, FP1dPotential


grid = Grid(10., 4)


@pytest.fixture(scope="session")
def epsilon_dist():
    return DielectricConstDist(
        ave_ele=[0.5]*3,
        ave_ion=[0.]*3,
        dist=ManualDist(10.0, 4, manual_dist=np.array([1.0]*4)))


@pytest.fixture(scope="session")
def gauss_charge_model():
    return GaussChargeModel(
        grids=Grids(xy_grids=XYGrids(lattice=np.array([[10., 0.], [0., 10.]]),
                                     num_grids=[4, 4]),
                    z_grid=grid),
        sigma=1.0,
        defect_z_pos_in_frac=0.0
    )


@pytest.fixture(scope="session")
def potential(epsilon_dist, gauss_charge_model):
    return CalcGaussChargePotential(epsilon_dist, gauss_charge_model).potential


@pytest.fixture(scope="session")
def fp_1d_potential():
    return FP1dPotential(grid=grid, potential=[0.1, 0.2, 0.3, 0.4])


@pytest.fixture(scope="session")
def slab_model(epsilon_dist, gauss_charge_model, potential, fp_1d_potential):
    return SlabModel(epsilon_dist, gauss_charge_model, potential, 2, fp_1d_potential)


def test_plot(gauss_charge_model):
    ax = plt.gca()
    gauss_charge_model.to_plot(ax)
    plt.show()


def test_json_file_mixin(gauss_charge_model, potential, fp_1d_potential, tmpdir):
    assert_json_roundtrip(gauss_charge_model, tmpdir)
    assert_json_roundtrip(potential, tmpdir)
    assert_json_roundtrip(fp_1d_potential, tmpdir)


def test_gauss_charge_model_charges(gauss_charge_model: GaussChargeModel):
    assert gauss_charge_model.periodic_charges[0][0][0] == 0.06349363593424097
    assert gauss_charge_model.farthest_z_from_defect == (2, 5.0)


def test_():
    z_grid = dict(length=10., num_grid=30)
    diele_const = DielectricConstDist(
        ave_ele=[1.]*3,
        ave_ion=[0.]*3,
        dist=GaussianDist(center=5.0, sigma=100000000.0, **z_grid))

    grids = Grids(XYGrids(np.array([[10., 0.], [0., 10.]]), [30]*2),
                  Grid(**z_grid))
    gauss = GaussChargeModel(grids,
                             sigma=0.3,
                             defect_z_pos_in_frac=0.5)

    calc_pot = CalcGaussChargePotential(
        dielectric_const=diele_const,
        gauss_charge_model=gauss)
    slab_model = SlabModel(epsilon=diele_const,
                           gauss_charge_model=gauss,
                           gauss_charge_potential=calc_pot.potential,
                           charge=2)
    print(slab_model.xy_potential)
    print(slab_model.gauss_charge_potential.xy_ave_potential)

    ProfilePlotter(plt, slab_model)
    plt.show()
