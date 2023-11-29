# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal
from vise.tests.helpers.assertion import assert_json_roundtrip

from pydefect_2d.dielectric.distribution import ManualDist, GaussianDist
from pydefect_2d.dielectric.dielectric_distribution import \
    DielectricConstDist
from pydefect_2d.three_d.grids import Grid, Grids, XYGrids
from pydefect_2d.three_d.slab_model_plotter import SlabModelPlotter
from pydefect_2d.three_d.slab_model import CalcGaussChargePotential, \
    GaussChargeModel, SlabModel
from pydefect_2d.one_d.potential import OneDFpPotential

grid = Grid(10., 4)


@pytest.fixture(scope="session")
def epsilon_dist():
    return DielectricConstDist(
        ave_ele=[0.5]*3,
        ave_ion=[0.]*3,
        dist=ManualDist(10.0, 4,
                        unscaled_in_plane_dist_=np.array([1.0]*4),
                        unscaled_out_of_plane_dist_=np.array([1.0]*4)))


@pytest.fixture(scope="session")
def gauss_charge_model():
    return GaussChargeModel(
        grids=Grids(xy_grids=XYGrids(lattice=np.array([[10., 0.], [0., 10.]]),
                                     num_grids=[4, 4]),
                    z_grid=grid),
        std_dev=1.0,
        gauss_pos_in_frac=0.0
    )


@pytest.fixture(scope="session")
def gauss_charge_potential(epsilon_dist, gauss_charge_model):
    return CalcGaussChargePotential(epsilon_dist, gauss_charge_model).potential


@pytest.fixture(scope="session")
def fp_1d_potential():
    return OneDFpPotential(grid=grid, potential=np.array([0.1, 0.2, 0.3, 0.4]))


@pytest.fixture(scope="session")
def slab_model(epsilon_dist, gauss_charge_model, gauss_charge_potential,
               fp_1d_potential):
    return SlabModel(epsilon_dist, gauss_charge_model, gauss_charge_potential,
                     2, fp_1d_potential)


def test_potential(gauss_charge_potential):
    actual = gauss_charge_potential.xy_ave_potential
    expected = np.array([2.3980723, -0.24543635, -1.90719959, -0.24543635])
    assert_almost_equal(actual, expected)


def test_plot(gauss_charge_model):
    ax = plt.gca()
    gauss_charge_model.to_plot(ax)
    plt.show()


def test_json_file_mixin(gauss_charge_model,
                         gauss_charge_potential,
                         fp_1d_potential,
                         tmpdir):
    # assert_json_roundtrip(gauss_charge_model, tmpdir)
    # assert_json_roundtrip(gauss_charge_potential, tmpdir)
    assert_json_roundtrip(fp_1d_potential, tmpdir)


def test_gauss_charge_model_charges(gauss_charge_model: GaussChargeModel):
    assert gauss_charge_model.periodic_charges[0][0][0] == 0.06349363593424097
    assert gauss_charge_model.farthest_z_from_defect == (2, 5.0)


def test_slab_model_plotter():
    z_grid = dict(length=10., num_grid=30)
    diele_const = DielectricConstDist(
        ave_ele=[1.]*3,
        ave_ion=[0.]*3,
        dist=GaussianDist(center=5.0,
                          in_plane_sigma=100000000.0,
                          out_of_plane_sigma=100000000.0,
                          **z_grid))

    grids = Grids(XYGrids(np.array([[10., 0.], [0., 10.]]), [30]*2),
                  Grid(**z_grid))
    gauss = GaussChargeModel(grids,
                             std_dev=0.3,
                             gauss_pos_in_frac=0.5)

    calc_pot = CalcGaussChargePotential(
        dielectric_const=diele_const,
        gauss_charge_model=gauss)
    potential = calc_pot.potential

    fp_potential = OneDFpPotential(grid=Grid(**z_grid),
                                   potential=np.array([1.0]*30))

    slab_model = SlabModel(diele_dist=diele_const,
                           gauss_charge_model=gauss,
                           gauss_charge_potential=potential,
                           charge_state=2,
                           fp_potential=fp_potential)

    SlabModelPlotter(plt, slab_model)
    plt.show()
