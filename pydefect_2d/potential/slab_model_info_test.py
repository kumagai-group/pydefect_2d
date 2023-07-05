# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest
from matplotlib import pyplot as plt
from vise.tests.helpers.assertion import assert_json_roundtrip

from pydefect_2d.potential.epsilon_distribution import \
    EpsilonGaussianDistribution
from pydefect_2d.potential.grids import Grid, Grids, XYGrids
from pydefect_2d.potential.plotter import ProfilePlotter
from pydefect_2d.potential.slab_model_info import CalcGaussChargePotential, \
    GaussChargeModel, SlabModel, FP1dPotential

grid = Grid(10., 4)


@pytest.fixture(scope="session")
def epsilon_dist():
    return EpsilonGaussianDistribution(
        grid=grid,
        ave_electronic_epsilon=[0.5]*3,
        ave_ionic_epsilon=[0.]*3,
        center=5.0, sigma=0.1)


@pytest.fixture(scope="session")
def gauss_model():
    return GaussChargeModel(
        Grids(xy_grids=XYGrids(lattice=np.array([[10., 0.], [0., 10.]]),
                               num_grids=[4, 4]),
              z_grid=grid),
        sigma=1.0,
        defect_z_pos_in_frac=0.0,
        epsilon_x=np.array([1.0] * 4),
        epsilon_y=np.array([1.0] * 4))


@pytest.fixture(scope="session")
def potential(epsilon_dist, gauss_model):
    return CalcGaussChargePotential(epsilon_dist, gauss_model).potential


@pytest.fixture(scope="session")
def fp_1d_potential():
    return FP1dPotential(grid=grid, potential=[0.1, 0.2, 0.3, 0.4])


@pytest.fixture(scope="session")
def slab_model(epsilon_dist, gauss_model, potential, fp_1d_potential):
    return SlabModel(epsilon_dist, gauss_model, potential, 2, fp_1d_potential)


def test_plot():
    charge_model = GaussChargeModel(
        Grids(xy_grids=XYGrids(lattice=np.array([[10., 0.], [0., 10.]]),
                               num_grids=[4, 4]),
              z_grid=Grid(10, 100)),
        sigma=1.0,
        defect_z_pos_in_frac=0.75,
        epsilon_x=np.array([1.0] * 100),
        epsilon_y=np.array([1.0] * 100))

    ax = plt.gca()
    charge_model.to_plot(ax)
    plt.show()


def test_json_file_mixin(gauss_model, potential, fp_1d_potential, tmpdir):
    assert_json_roundtrip(gauss_model, tmpdir)
    assert_json_roundtrip(potential, tmpdir)
    assert_json_roundtrip(fp_1d_potential, tmpdir)


def test_gauss_charge_model_charges(gauss_model: GaussChargeModel):
    assert gauss_model.charges[0][0][0] == 0.06349363593424097
    assert gauss_model.farthest_z_from_defect == (2, 5.0)


def test_():
    num_grid = 30
    grid_z = Grid(10., num_grid)
    eps = EpsilonGaussianDistribution(
        grid=grid_z,
        ave_electronic_epsilon=[0.]*3,
        ave_ionic_epsilon=[0.]*3,
        center=5.0, sigma=100000000)

    grids = Grids(XYGrids(np.array([[10., 0.], [0., 10.]]), [num_grid]*2), grid_z)
    gauss = GaussChargeModel(grids,
                             sigma=0.3,
                             defect_z_pos_in_frac=0.5,
                             epsilon_x=np.array([1.0]*num_grid),
                             epsilon_y=np.array([1.0]*num_grid))

    calc_pot = CalcGaussChargePotential(
        epsilon=eps,
        gauss_charge_model=gauss)
    slab_model = SlabModel(epsilon=eps,
                           gauss_charge_model=gauss,
                           gauss_charge_potential=calc_pot.potential,
                           charge=2)
    print(slab_model.xy_potential)
    print(slab_model.gauss_charge_potential.xy_ave_potential)

    ProfilePlotter(plt, slab_model)
    plt.show()
